"""
Motor Segment Transition Analysis

Corresponds to Figure 5 in the paper.

This module visualizes state transition properties in two specific segments of
unwrapped angular phase data. It includes:

1. State detection using kernel density estimation (KDE) on smoothed phase data
2. Segmented analysis comparing two halves of each data segment
3. State occupation histograms
4. Transition rate calculations with error estimation

The output includes publication-quality figures showing phase dynamics and
transition statistics for comparison across different conditions.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity


# ============================================================================
# OPTICAL PARAMETERS (capital letters - adjust as needed)
# ============================================================================
SAMPLING_FREQUENCY = 250000  # Hz
DATA_FOLDER = "./data/"  # Path to data folder
ANGLE_UNITS = "radians"  # Units for phase data (radians or degrees)

# KDE parameters for state detection
KDE_KERNEL = "gaussian"
KDE_BANDWIDTH_DEFAULT = 0.012 / (2 * np.pi)  # Default bandwidth for KDE in revolutions
KDE_GRID_POINTS = 1000

# Peak detection parameters
MIN_PEAK_DISTANCE_MOTOR2 = int(0.1 / 0.02)  # Based on state spacing
MIN_PEAK_DISTANCE_MOTOR4 = int(0.2 / 0.02)
PEAK_HEIGHT_THRESHOLD_MOTOR4 = 0.5

# Figure parameters
SCATTER_POINT_SIZE = 0.01
STATE_LINE_WIDTH = 0.5
FIGURE_DPI = 300
FIGURE_FORMAT = "png"


def load_phase_data_from_file(name, folder):
    """
    Load unwrapped phase and step boundary data from file.

    Loads preprocessed phase data in .npz format containing unwrapped phase,
    smoothed phase, detected steps, and time array.

    Parameters
    ----------
    data_file_path : str
        Path to .npz file containing phase data

    Returns
    -------
    time_array : ndarray
        Time values (in seconds)
    phi_unwrapped : ndarray
        Unwrapped phase trajectory (in radians)
    phi_smoothed : ndarray
        Smoothed phase (state-based)
    state_levels : ndarray
        Detected state levels
    step_boundaries : ndarray
        Sample indices of state boundaries
    """
    folderlist = os.listdir(folder)
    print(folderlist)
    for f in folderlist:
        if name in f and f.endswith(".npy"):
            data = np.load(folder + f)
            if data.ndim == 1:
                phiu = data
            elif data.ndim == 2:
                if data.shape[0] == 2:
                    xt = data[0]
                    phiu = data[1]
                elif data.shape[1] == 2:
                    phiu = data[:, 1]
                    xt = data[:, 0]
        if name in f and f.endswith(".npz") and "cutoff8" in f:
            trans = np.load(folder + f)
            xbound = trans["peaks"]
            m = trans["m"]
            phi_smooth = np.zeros_like(phiu)
            for i in range(len(xbound) - 1):
                phi_smooth[xbound[i] : xbound[i + 1]] = m[i]
            return (
                xt,
                phiu / 180 * np.pi,
                phi_smooth / 180 * np.pi,
                m / 180 * np.pi,
                xbound,
            )
    return xt, phiu / 180 * np.pi, None, None, None


def detect_states_via_kde(state_data, bandwidth=None, min_peak_distance=50):
    """
    Detect distinct states using kernel density estimation.

    Applies KDE to state values and identifies peaks as distinct states.

    Parameters
    ----------
    state_data : array_like
        State level values (1D array)
    bandwidth : float, optional
        KDE bandwidth (default: KDE_BANDWIDTH_DEFAULT)
    min_peak_distance : int, optional
        Minimum samples between peaks

    Returns
    -------
    kde : KernelDensity
        Fitted KDE object
    grid_values : ndarray
        Grid where density was evaluated
    density : ndarray
        Probability density
    unique_states : ndarray
        Detected state levels (peak locations)
    peak_indices : ndarray
        Indices of peaks in grid
    """
    if bandwidth is None:
        bandwidth = KDE_BANDWIDTH_DEFAULT

    # Fit KDE to state data
    kde = KernelDensity(kernel=KDE_KERNEL, bandwidth=bandwidth)
    kde.fit(state_data[:, None])

    # Evaluate on grid
    grid_values = np.linspace(np.min(state_data), np.max(state_data), KDE_GRID_POINTS)
    log_density = kde.score_samples(grid_values[:, None])
    density = np.exp(log_density)

    # Find peaks
    peak_indices, _ = find_peaks(density, distance=min_peak_distance)
    unique_states = grid_values[peak_indices]

    return kde, grid_values, density, unique_states, peak_indices


def map_states_to_detected(data_array, unique_states):
    """
    Map data values to nearest detected state.

    For each value in data, find the closest state in unique_states.

    Parameters
    ----------
    data_array : array_like
        Data values to map
    unique_states : array_like
        Detected state levels

    Returns
    -------
    state_indices : ndarray
        Index of nearest state for each data point
    state_values : ndarray
        Actual state values
    """
    state_indices = np.array(
        [np.argmin(np.abs(val - unique_states)) for val in data_array]
    )

    state_values = unique_states[state_indices]

    return state_indices, state_values


def analyze_segment_transition_rates(
    phase_indices,
    state_indices,
    half_length,
    from_state=None,
    to_state=None,
    sampling_freq=SAMPLING_FREQUENCY,
):
    """
    Analyze transition rates between two halves of a segment.

    Computes transition rates (first-order kinetics) for transitions from
    one state to another in two halves of the data.

    Parameters
    ----------
    phase_indices : array_like
        Time or phase indices for data points
    state_indices : ndarray
        State index for each data point
    half_length : int
        Length of each half
    from_state : int, optional
        Source state index
    to_state : int, optional
        Target state index
    sampling_freq : float, optional
        Sampling frequency

    Returns
    -------
    results : dict
        Dictionary with:
        - 'rate_1': Transition rate in first half (s^-1)
        - 'rate_2': Transition rate in second half (s^-1)
        - 'error_1': Error in first half rate
        - 'error_2': Error in second half rate
        - 'count_1': Number of transitions in first half
        - 'count_2': Number of transitions in second half
        - 'dwell_1': Time spent in source state (first half)
        - 'dwell_2': Time spent in source state (second half)
    """
    # Split into halves
    first_half = state_indices[:half_length]
    second_half = state_indices[half_length:]

    # Time spent in source state
    dwell_time_1 = np.sum(first_half == from_state) / sampling_freq
    dwell_time_2 = np.sum(second_half == from_state) / sampling_freq

    # Count transitions
    transition_count_1 = np.sum(
        (first_half[:-1] == from_state) & (first_half[1:] == to_state)
    )
    transition_count_2 = np.sum(
        (second_half[:-1] == from_state) & (second_half[1:] == to_state)
    )

    # Calculate rates
    rate_1 = transition_count_1 / dwell_time_1 if dwell_time_1 > 0 else 0
    rate_2 = transition_count_2 / dwell_time_2 if dwell_time_2 > 0 else 0

    # Error (Poisson)
    error_1 = np.sqrt(transition_count_1) / dwell_time_1 if dwell_time_1 > 0 else 0
    error_2 = np.sqrt(transition_count_2) / dwell_time_2 if dwell_time_2 > 0 else 0

    return {
        "rate_1": rate_1,
        "rate_2": rate_2,
        "error_1": error_1,
        "error_2": error_2,
        "count_1": transition_count_1,
        "count_2": transition_count_2,
        "dwell_1": dwell_time_1,
        "dwell_2": dwell_time_2,
    }


def plot_segment_analysis(
    time_array,
    phi_unwrapped,
    phi_smoothed,
    state_levels,
    step_boundaries,
    time_range,
    unique_states=None,
    state_indices=None,
    title="Phase Segment Analysis",
):
    """
    Create three-panel plot of phase segment analysis.

    Shows: (1) Raw and smoothed phase, (2) State occupation histogram,
    (3) Transition rate statistic.

    Parameters
    ----------
    time_array : ndarray
        Time values
    phi_unwrapped : ndarray
        Unwrapped phase
    phi_smoothed : ndarray
        Smoothed phase
    state_levels : ndarray
        State level values
    step_boundaries : ndarray
        Step boundary indices
    time_range : [float, float]
        Time range to plot [t_min, t_max]
    unique_states : ndarray, optional
        Detected unique state values
    state_indices : ndarray, optional
        State index for each point
    title : str, optional
        Figure title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : array of Axes
        Subplot axes [ax_phase, ax_occupation, ax_rate]
    """
    # Select time range
    time_mask = (time_array >= time_range[0]) & (time_array <= time_range[1])
    t_seg = time_array[time_mask]
    phi_seg = phi_unwrapped[time_mask]
    phi_smooth_seg = phi_smoothed[time_mask]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Phase time series
    ax1 = axes[0]
    half_len = len(t_seg) // 2

    ax1.scatter(
        t_seg[:half_len],
        phi_seg[:half_len],
        s=SCATTER_POINT_SIZE,
        alpha=0.5,
        label="First half (data)",
    )
    ax1.scatter(
        t_seg[half_len:],
        phi_seg[half_len:],
        s=SCATTER_POINT_SIZE,
        alpha=0.5,
        label="Second half (data)",
    )

    if state_indices is not None and unique_states is not None:
        state_values = unique_states[state_indices[time_mask]]
        ax1.plot(
            t_seg,
            state_values / (2 * np.pi),
            color="red",
            linewidth=STATE_LINE_WIDTH,
            label="Detected states",
        )

    ax1.set_xlabel("Time (s)", labelpad=5)
    ax1.set_ylabel("Phase (revolutions)", labelpad=5)
    ax1.set_title("Phase Dynamics")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: State occupation (bifold histogram)
    ax2 = axes[1]
    if state_indices is not None:
        state_indices_seg = state_indices[time_mask]
        unique_idx = np.unique(state_indices_seg)

        count_1 = np.bincount(state_indices_seg[:half_len], minlength=len(unique_idx))
        count_2 = np.bincount(state_indices_seg[half_len:], minlength=len(unique_idx))
        count_1 = count_1 / np.sum(count_1)
        count_2 = count_2 / np.sum(count_2)

        states_range = range(1, len(unique_idx) + 1)
        ax2.bar([x - 0.2 for x in states_range], count_1, width=0.4, label="First half")
        ax2.bar(
            [x + 0.2 for x in states_range], -count_2, width=0.4, label="Second half"
        )

        ax2.set_xticks(states_range)
        ax2.set_xlabel("State", labelpad=5)
        ax2.set_ylabel("Occupancy (normalized)", labelpad=5)
        ax2.set_title("State Occupation")
        ax2.legend(fontsize=8)
        ax2.axhline(0, color="k", linestyle="-", linewidth=0.5)

    # Panel 3: Transition rate (placeholder)
    ax3 = axes[2]
    ax3.text(
        0.5,
        0.5,
        "Transition Rate\n(Computed separately)",
        ha="center",
        va="center",
        transform=ax3.transAxes,
        fontsize=12,
    )
    ax3.set_title("Transition Statistics")
    ax3.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig, axes


def create_figure5_complete(
    name_motor2,
    name_motor4,
    folder,
    segment_motor2=(1038, 1055),
    segment_motor4=(408, 411.5),
):
    """
    Create complete Figure 5 with both motor segments.

    Parameters
    ----------
    data_file_motor2 : str
        Path to Motor2 data file
    data_file_motor4 : str
        Path to Motor4 data file
    segment_motor2 : tuple of float
        Time range for Motor2 segment (t_min, t_max) in seconds
    segment_motor4 : tuple of float
        Time range for Motor4 segment (t_min, t_max) in seconds

    Returns
    -------
    figures : dict
        Dictionary with figure objects for each segment
    """
    figures = {}

    try:
        # Load Motor2 data
        print("Loading Motor2 data...")
        (time_m2, phi_m2, phi_smooth_m2, state_m2, bounds_m2) = (
            load_phase_data_from_file(name_motor2, folder)
        )

        # Detect states
        kde_m2, grid_m2, dens_m2, unique_m2, peaks_m2 = detect_states_via_kde(
            state_m2, min_peak_distance=MIN_PEAK_DISTANCE_MOTOR2
        )
        state_indices_m2, _ = map_states_to_detected(phi_smooth_m2, unique_m2)

        # Create figure
        fig_m2, ax_m2 = plot_segment_analysis(
            time_m2,
            phi_m2,
            phi_smooth_m2,
            state_m2,
            bounds_m2,
            segment_motor2,
            unique_m2,
            state_indices_m2,
            title="Figure 5 - Motor2 Segment",
        )
        figures["motor2"] = (fig_m2, ax_m2)

        # Load Motor4 data
        print("Loading Motor4 data...")
        (time_m4, phi_m4, phi_smooth_m4, state_m4, bounds_m4) = (
            load_phase_data_from_file(name_motor4, folder)
        )

        # Detect states
        kde_m4, grid_m4, dens_m4, unique_m4, peaks_m4 = detect_states_via_kde(
            state_m4,
            min_peak_distance=MIN_PEAK_DISTANCE_MOTOR4,
            bandwidth=KDE_BANDWIDTH_DEFAULT,
        )
        state_indices_m4, _ = map_states_to_detected(phi_smooth_m4, unique_m4)

        # Create figure
        fig_m4, ax_m4 = plot_segment_analysis(
            time_m4,
            phi_m4,
            phi_smooth_m4,
            state_m4,
            bounds_m4,
            segment_motor4,
            unique_m4,
            state_indices_m4,
            title="Figure 5 - Motor4 Segment",
        )
        figures["motor4"] = (fig_m4, ax_m4)

        return figures

    except Exception as e:
        print(f"Error creating Figure 5: {e}")
        return {}

