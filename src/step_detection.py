"""Step Detection Module

This module implements advanced step detection algorithms using chi-squared weighted
filtering and iterative segment merging. The algorithms are optimized for high-performance
computation using Numba JIT compilation.

Main functions:
    - apply_chi2_filter: Apply chi-squared weighted filtering to signal data
    - detect_steps: Detect steps in noisy data using hierarchical segment merging
"""

import numpy as np
import numba
from numba import njit


@njit(parallel=True, fastmath=True)
def chi2_weighted_filter_flat_steps(signal_data, noise_std_dev):
    """Chi-squared weighted filter for signal with flat step levels.

    Applies multi-scale sliding window filtering where each segment is modeled as
    a constant level. Weights are computed from chi-squared goodness-of-fit, providing
    robust denoising for step-like signals.

    Args:
        signal_data (np.ndarray): 1D signal array to filter
        noise_std_dev (float): Standard deviation of measurement noise

    Returns:
        np.ndarray: Filtered signal data
    """
    data_length = len(signal_data)
    window_scales = np.linspace(10, 1000, 10)
    window_scales = window_scales.astype(np.int32)
    filtered_signal = np.zeros(data_length, dtype=np.float64)
    weight_accumulation = np.zeros(data_length, dtype=np.float64)

    # Process each window scale in parallel
    for window_idx in numba.prange(len(window_scales)):
        window_size = window_scales[window_idx]

        # Sliding window over signal
        for start_idx in range(data_length - window_size + 1):
            window_data = signal_data[start_idx : start_idx + window_size]
            level_estimate = np.sum(window_data) / window_size

            # Chi-squared goodness-of-fit for constant model
            chi2_metric = np.sum((window_data - level_estimate) ** 2) / (
                window_size * noise_std_dev**2
            )
            weight = np.exp(-chi2_metric)

            # Accumulate weighted contributions
            filtered_signal[start_idx : start_idx + window_size] += (
                level_estimate * weight
            )
            weight_accumulation[start_idx : start_idx + window_size] += weight

    return filtered_signal / weight_accumulation


@njit
def detect_steps_algorithm_core(signal_array, min_step_amplitude=10.0):
    """Core step detection algorithm using hierarchical segment merging.

    Implements an iterative algorithm that:
    1. Initializes segments by coarse windowing
    2. Computes mean level for each segment
    3. Merges adjacent segments with amplitude difference below threshold
    4. Reconstructs signal with detected step levels

    Args:
        signal_array (np.ndarray): 1D signal to analyze
        min_step_amplitude (float): Minimum amplitude threshold for step retention

    Returns:
        tuple: (reconstructed_signal, segment_boundaries, segment_levels)
    """
    data_length = len(signal_array)
    max_segments = data_length // 5 + 2

    # Segment tracking arrays
    segment_boundaries = np.empty(max_segments, dtype=np.int64)
    segment_levels = np.empty(max_segments, dtype=signal_array.dtype)
    segment_valid = np.ones(max_segments, dtype=np.bool_)
    segment_next_link = np.empty(max_segments, dtype=np.int64)

    # Phase 1: Initial coarse segmentation
    segment_count = 0
    for boundary_idx in range(0, data_length, 5):
        segment_boundaries[segment_count] = boundary_idx
        segment_count += 1
    segment_boundaries[segment_count] = data_length
    segment_count += 1

    # Phase 2: Setup linked list structure
    for seg_idx in range(segment_count - 1):
        segment_next_link[seg_idx] = seg_idx + 1
    segment_next_link[segment_count - 1] = -1  # Terminator

    # Phase 3: Compute initial segment levels
    for seg_idx in range(segment_count - 1):
        start_pos = segment_boundaries[seg_idx]
        end_pos = segment_boundaries[seg_idx + 1]
        segment_levels[seg_idx] = np.mean(signal_array[start_pos:end_pos])

    # Phase 4: Iterative merging of adjacent segments
    current_seg = 0
    while current_seg != -1:
        next_seg = segment_next_link[current_seg]
        if next_seg == -1:
            break

        # Check if amplitude difference is below threshold
        amplitude_diff = abs(segment_levels[next_seg] - segment_levels[current_seg])
        if amplitude_diff < min_step_amplitude:
            # Merge: mark next segment as invalid
            segment_valid[next_seg] = False
            segment_next_link[current_seg] = segment_next_link[next_seg]

            # Recompute merged segment level
            merge_right_link = segment_next_link[next_seg]
            right_boundary = (
                segment_boundaries[merge_right_link]
                if merge_right_link != -1
                else data_length
            )
            left_boundary = segment_boundaries[current_seg]
            segment_levels[current_seg] = np.mean(
                signal_array[left_boundary:right_boundary]
            )
            # Retry with same current_seg (new next_seg)
        else:
            current_seg = next_seg

    # Phase 5: Build output
    reconstructed_signal = np.empty_like(signal_array)
    output_boundaries = np.empty(max_segments, dtype=np.int64)
    output_levels = np.empty(max_segments, dtype=signal_array.dtype)
    output_count = 0

    # Collect valid segments
    current_seg = 0
    while current_seg != -1:
        output_boundaries[output_count] = segment_boundaries[current_seg]
        if segment_next_link[current_seg] != -1:
            output_levels[output_count] = segment_levels[current_seg]
        output_count += 1
        current_seg = segment_next_link[current_seg]

    output_boundaries[output_count] = data_length
    output_count += 1

    # Reconstruct signal with detected levels
    for seg_idx in range(output_count - 1):
        start_pos = output_boundaries[seg_idx]
        end_pos = output_boundaries[seg_idx + 1]
        reconstructed_signal[start_pos:end_pos] = output_levels[seg_idx]

    return (
        reconstructed_signal,
        output_boundaries[:output_count],
        output_levels[: output_count - 1],
    )


if __name__ == "__main__":
    # Example usage from unwrapped angle datas 
    # First select manually a long dwell from the unwrapped phi signal, then apply the step detection algorithm
    # Compute the noise standard deviation from the long dwell, then apply the chi2 weighted filter, and finally apply the step detection algorithm
    phi_unwrapped = np.load("phi_unwrapped.npy")
    long_dwell_from_phi = np.load("long_dwell_from_phi.npy")
    noise_std_dev = np.std(long_dwell_from_phi)
    phi_filtered = chi2_weighted_filter_flat_steps(
        phi_unwrapped, noise_std_dev=noise_std_dev
    )
    step_signal, boundaries, levels = detect_steps_algorithm_core(phi_filtered)
    print("Detected step boundaries:", boundaries)
    print("Detected step levels:", levels)
