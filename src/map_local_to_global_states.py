"""State Mapping Module

This module implements algorithms for mapping local step-detected states to global
states identified through kernel density estimation of the full angular distribution.

The workflow:
1. Takes unwrapped azimuthal angle data (phi_unwrapped) and step detection results
2. Identifies global states using kernel density estimation on the full phi distribution
3. Maps each discrete step from step detection to the closest global state
4. Handles periodic boundary conditions (2π wrapping) for angular data

Key functions:
    - find_global_states_kde: Identify global states via KDE peak detection
    - map_steps_to_global_states: Map local steps to global states with periodicity handling
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import signal


def find_global_states_kde(
    phi_unwrapped, bandwidth=0.012, prominence_threshold=0.01, min_peak_distance_deg=5
):
    """Identify global states using kernel density estimation on unwrapped phi data.

    Applies Gaussian kernel density estimation to the full distribution of phi_unwrapped
    and detects peaks to identify global angular states.

    Args:
        phi_unwrapped (np.ndarray): Unwrapped azimuthal angle data (radians)
        bandwidth (float): KDE bandwidth parameter
        prominence_threshold (float): Minimum peak prominence for detection
        min_peak_distance_deg (float): Minimum angular separation between peaks (degrees)

    Returns:
        np.ndarray: Global state angles (radians, in [0, 2π))
    """
    # Wrap angles to [0, 2π) for density estimation
    phi_wrapped = phi_unwrapped % (2 * np.pi)
    phi_wrapped = phi_wrapped[:, np.newaxis]

    # Fit kernel density estimator
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(phi_wrapped)

    # Evaluate density on fine grid
    angle_range_start = 0
    angle_range_end = 2 * np.pi
    angle_step = 0.01
    angle_grid = np.arange(angle_range_start, angle_range_end, angle_step)
    log_density = kde.score_samples(angle_grid[:, np.newaxis])
    density = np.exp(log_density)

    # Find peaks in density
    min_distance_samples = int(min_peak_distance_deg / 180 * np.pi / angle_step)
    peaks, _ = signal.find_peaks(
        density, prominence=prominence_threshold, distance=min_distance_samples
    )

    # Convert peak indices to angles
    global_states = angle_grid[peaks]

    return global_states


def map_steps_to_global_states(
    global_states, step_levels, step_boundaries, output_filename=None
):
    """Map discrete step levels to closest global states with periodic boundary handling.

    For each step level, finds the closest global state considering the periodic
    nature of angular data (2π wrapping). Removes duplicate mappings and saves results.

    Args:
        global_states (np.ndarray): Global state angles from KDE (radians)
        step_levels (np.ndarray): Step level values from step detection (radians)
        step_boundaries (np.ndarray): Step boundary indices
        output_filename (str): Optional filename for saving results (NPZ format)

    Returns:
        tuple: (filtered_boundaries, mapped_levels) - boundaries and mapped levels
    """
    mapped_levels = []
    filtered_boundaries = list(step_boundaries.copy())

    for step_level in step_levels:
        # Handle periodic boundary conditions
        adjusted_level = step_level
        period_offset = 0

        # Adjust level to be within reasonable range for comparison
        while adjusted_level < 0:
            adjusted_level += 2 * np.pi
            period_offset += 1
        while adjusted_level > 2 * np.pi:
            adjusted_level -= 2 * np.pi
            period_offset -= 1

        # Find closest global state
        distances = np.abs(global_states - adjusted_level)
        closest_idx = np.argmin(distances)
        closest_global = global_states[closest_idx]

        # Apply period offset back
        mapped_level = closest_global - period_offset * 2 * np.pi
        mapped_levels.append(mapped_level)

    # Remove consecutive duplicates (same global state)
    index = 0
    while index < len(mapped_levels) - 1:
        if mapped_levels[index + 1] == mapped_levels[index]:
            mapped_levels.pop(index + 1)
            filtered_boundaries.pop(index + 1)
            index -= 1  # Recheck current position
        index += 1

    # Convert to degrees for output
    mapped_levels_deg = np.array(mapped_levels) * 180 / np.pi

    # Save results if filename provided
    if output_filename is not None:
        np.savez(
            output_filename,
            boundaries=filtered_boundaries,
            levels_radians=np.array(mapped_levels),
            levels_degrees=mapped_levels_deg,
        )

    return filtered_boundaries, mapped_levels


def process_state_mapping(
    phi_unwrapped,
    step_levels,
    step_boundaries,
    kde_bandwidth=0.012,
    output_filename=None,
):
    """Complete pipeline for mapping local steps to global states.

    Combines global state detection and step mapping into a single function.

    Args:
        phi_unwrapped (np.ndarray): Unwrapped azimuthal angle data
        step_levels (np.ndarray): Step levels from step detection
        step_boundaries (np.ndarray): Step boundaries from step detection
        kde_bandwidth (float): Bandwidth for KDE global state detection
        output_filename (str): Optional output filename

    Returns:
        tuple: (global_states, filtered_boundaries, mapped_levels)
    """
    # Find global states
    global_states = find_global_states_kde(phi_unwrapped, bandwidth=kde_bandwidth)

    # Map steps to global states
    filtered_boundaries, mapped_levels = map_steps_to_global_states(
        global_states, step_levels, step_boundaries, output_filename
    )

    return global_states, filtered_boundaries, mapped_levels

