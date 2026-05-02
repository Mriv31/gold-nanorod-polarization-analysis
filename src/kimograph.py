"""
Kimograph generation for unwrapped angular trajectories.

This module provides a single reusable function to compute a kimograph by sliding
a window over an unwrapped phase signal and estimating the local phase density
using kernel density estimation. Detected peaks in the density correspond to
significant periodic states in the phase signal.

Dependencies: numpy, scipy.signal, sklearn.neighbors
"""

import numpy as np
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity


# Default analysis parameters
SAMPLING_FREQUENCY = 250000  # Hz
DEFAULT_WINDOW = 7500  # samples
DEFAULT_SHIFT = 7500  # samples
DEFAULT_MAX_DURATION_SECONDS = 200.0  # seconds
DEFAULT_BANDWIDTH = 0.012 / (2 * np.pi)
DEFAULT_RESOLUTION = 0.01 / (2 * np.pi)
DEFAULT_PROMINENCE = 0.5
DEFAULT_STATE_DISTANCE = int((4 / 180) * np.pi / DEFAULT_RESOLUTION)


def compute_kimograph(
    phi_unwrapped,
    window=DEFAULT_WINDOW,
    shift=DEFAULT_SHIFT,
    sampling_frequency=SAMPLING_FREQUENCY,
    max_duration_seconds=DEFAULT_MAX_DURATION_SECONDS,
    bandwidth=DEFAULT_BANDWIDTH,
    resolution=DEFAULT_RESOLUTION,
    prominence=DEFAULT_PROMINENCE,
    min_peak_distance=None,
):
    """
    Compute a kimograph from an unwrapped phase trajectory.

    The function slides a fixed-size window along the input trajectory, applies
    kernel density estimation to the normalized phase values within each window,
    and detects peaks in the resulting density estimate.

    Parameters
    ----------
    phi_unwrapped : array_like
        Unwrapped phase trajectory in revolutions or normalized units.
    window : int, optional
        Window size in samples.
    shift : int, optional
        Shift increment between successive windows in samples.
    sampling_frequency : float, optional
        Sampling frequency in Hz.
    max_duration_seconds : float, optional
        Maximum duration to process in seconds.
    bandwidth : float, optional
        Bandwidth used by the KDE estimator.
    resolution : float, optional
        Evaluation spacing for the KDE grid.
    prominence : float, optional
        Minimum peak prominence for state detection.
    min_peak_distance : int, optional
        Minimum distance between peaks in grid points.

    Returns
    -------
    dict
        Dictionary containing:
        - x_grid_list : list of ndarray, KDE evaluation grids for each window
        - peak_index_list : list of ndarray, detected peak indices for each window
        - density_list : list of ndarray, KDE density values for each window
        - prominence_list : list of ndarray, peak prominences for each window
        - window_start_list : list of int, start indices for each window
        - window_stop_list : list of int, stop indices for each window
    """
    phi = np.asarray(phi_unwrapped, dtype=float)
    if phi.ndim != 1:
        raise ValueError("phi_unwrapped must be a one-dimensional array.")

    if window <= 0 or shift <= 0:
        raise ValueError("window and shift must be positive integers.")

    if min_peak_distance is None:
        min_peak_distance = DEFAULT_STATE_DISTANCE
    elif min_peak_distance < 1:
        min_peak_distance = 1

    max_samples = min(len(phi), int(max_duration_seconds * sampling_frequency))
    if window > max_samples:
        raise ValueError(
            "window is larger than the available data length for the requested duration."
        )

    x_grid_list = []
    peak_index_list = []
    density_list = []
    prominence_list = []
    window_start_list = []
    window_stop_list = []

    start = 0
    stop = window
    n_windows = int((max_samples - window) // shift) + 1

    for _ in range(n_windows):
        segment = phi[start:stop] % 1.0  # IN REVOLUTIONS
        segment = segment[:, np.newaxis]

        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(segment)

        x_grid = np.arange(0.0, 1.0, resolution)
        log_density = kde.score_samples(x_grid[:, np.newaxis])
        density = np.exp(log_density)

        peaks, properties = find_peaks(
            density,
            prominence=prominence,
            distance=min_peak_distance,
        )

        x_grid_list.append(x_grid)
        peak_index_list.append(peaks)
        density_list.append(density)
        prominence_list.append(properties.get("prominences", np.array([], dtype=float)))
        window_start_list.append(start)
        window_stop_list.append(stop)

        start += shift
        if stop >= max_samples:
            break
        stop = min(stop + shift, max_samples)

    return {
        "x_grid_list": x_grid_list,
        "peak_index_list": peak_index_list,
        "density_list": density_list,
        "prominence_list": prominence_list,
        "window_start_list": window_start_list,
        "window_stop_list": window_stop_list,
    }
