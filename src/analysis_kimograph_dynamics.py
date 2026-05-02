"""Kimograph-based Phase Dynamics Analysis

Corresponds to Figure 6 in the paper.

Builds  kimograph visualizations, phase histograms, and trace panels
from unwrapped phase trajectories. Performs phase dynamics and state analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import find_peaks
from .kimograph import compute_kimograph


# Default plotting and analysis parameters
SAMPLING_FREQUENCY = 250000  # Hz
DEFAULT_WINDOW = 7500  # samples
DEFAULT_SHIFT = 7500  # samples
DEFAULT_TIME_SCALE = 1.0 / SAMPLING_FREQUENCY
DEFAULT_KDE_BANDWIDTH = 0.012 / (2 * np.pi)
DEFAULT_KDE_RESOLUTION = 0.01 / (2 * np.pi)
DEFAULT_PROMINENCE = 0.5
DEFAULT_STATE_DISTANCE = int((4 / 180) * np.pi / DEFAULT_KDE_RESOLUTION)
DEFAULT_BIN_WIDTH = 1.0 / 600.0


def load_phase_and_state_data(folder, name):
    """
    Load phase trajectory data and filtered state information.

    Parameters
    ----------
    folder : str
        Directory containing phase and state data files.
    name : str
        Base name to search for in file names.

    Returns
    -------
    tuple
        (time_array, phi_unwrapped, phi_smooth, state_levels, step_boundaries)
    """
    xt = None
    phiu = None
    phi_smooth = None
    state_levels = None
    step_boundaries = None

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        if name in filename and filename.endswith(".npy"):
            data = np.load(filepath)
            if data.ndim == 1:
                phiu = data
            elif data.ndim == 2:
                if data.shape[0] == 2:
                    xt = data[0]
                    phiu = data[1]
                elif data.shape[1] == 2:
                    xt = data[:, 0]
                    phiu = data[:, 1]

        if name in filename and filename.endswith(".npz") and "cutoff8" in filename:
            trans = np.load(filepath)
            step_boundaries = trans["peaks"]
            state_levels = trans["m"]

            if phiu is None:
                raise ValueError(
                    "Phase trajectory .npy file must be present before the .npz file."
                )

            phi_smooth = np.zeros_like(phiu)
            for idx in range(len(step_boundaries) - 1):
                phi_smooth[step_boundaries[idx] : step_boundaries[idx + 1]] = (
                    state_levels[idx]
                )

            break

    if phiu is None:
        raise FileNotFoundError(
            f"Could not find unwrapped phase file for name '{name}' in {folder}"
        )

    if xt is None:
        xt = np.arange(len(phiu)) * DEFAULT_TIME_SCALE

    return (
        xt,
        phiu / 180.0 * np.pi,
        (phi_smooth / 180.0 * np.pi if phi_smooth is not None else None),
        (state_levels / 180.0 * np.pi if state_levels is not None else None),
        step_boundaries,
    )


def log_formatter(x, pos):
    return f"{np.exp(x):.1f}"


def plot_kimograph(
    ax,
    phi_unwrapped,
    kimograph_result,
    shift=DEFAULT_SHIFT,
    sampling_frequency=SAMPLING_FREQUENCY,
    point_size=2,
    cmap="viridis",
    vmin=-1,
    vmax=3,
):
    """
    Plot the kimograph peak positions for each window.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to draw the kimograph on.
    phi_unwrapped : array_like
        Unwrapped phase trajectory.
    kimograph_result : dict
        Result of compute_kimograph.
    shift : int, optional
        Window shift in samples.
    sampling_frequency : float, optional
        Sampling frequency in Hz.
    point_size : float, optional
        Marker size for peak points.
    cmap : str, optional
        Colormap for density scaling.
    vmin : float, optional
        Minimum log-density for the colormap.
    vmax : float, optional
        Maximum log-density for the colormap.

    Returns
    -------
    PathCollection
        Scatter plot handle for the peak points.
    """
    x_grid_list = kimograph_result["x_grid_list"]
    peak_index_list = kimograph_result["peak_index_list"]
    density_list = kimograph_result["density_list"]

    time_per_shift = shift / sampling_frequency

    scatter_handles = []
    for window_idx, peaks in enumerate(peak_index_list):
        if len(peaks) == 0:
            continue

        x_positions = np.full(len(peaks), (window_idx + 0.5) * time_per_shift)
        y_values = x_grid_list[window_idx][peaks]
        colors = np.log(density_list[window_idx][peaks])

        scatter = ax.scatter(
            x_positions,
            y_values,
            c=colors,
            cmap=cmap,
            s=point_size,
            vmin=vmin,
            vmax=vmax,
        )
        scatter_handles.append(scatter)

    ax.set_ylabel("Peak position (revs)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)

    return scatter_handles[-1] if scatter_handles else None


def plot_phase_histograms(
    ax,
    phase_signal,
    kimograph_result,
    window_index_1,
    window_index_2,
    bin_width=DEFAULT_BIN_WIDTH,
    density_color_1="red",
    density_color_2="black",
    kde_color="blue",
):
    """
    Plot two phase histograms and their corresponding KDE fits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object for the histogram.
    phase_signal : array_like
        Raw phase trajectory.
    kimograph_result : dict
        Result of compute_kimograph.
    window_index_1 : int
        First window index to plot.
    window_index_2 : int
        Second window index to plot.
    bin_width : float, optional
        Histogram bin width in phase units.
    density_color_1 : str, optional
        Bar color for first window.
    density_color_2 : str, optional
        Bar color for second window.
    kde_color : str, optional
        Color for KDE fit curves.
    """
    start_list = kimograph_result["window_start_list"]
    stop_list = kimograph_result["window_stop_list"]
    x_grid_list = kimograph_result["x_grid_list"]
    density_list = kimograph_result["density_list"]
    peak_index_list = kimograph_result["peak_index_list"]

    bins = np.arange(0.0, 1.0 + bin_width, bin_width)

    hist_1, edges_1 = np.histogram(
        phase_signal[start_list[window_index_1] : stop_list[window_index_1]] % 1.0,
        bins=bins,
        density=True,
    )
    hist_2, edges_2 = np.histogram(
        phase_signal[start_list[window_index_2] : stop_list[window_index_2]] % 1.0,
        bins=bins,
        density=True,
    )

    ax.bar(edges_1[:-1], hist_1, width=bin_width, color=density_color_1, alpha=0.6)
    ax.bar(edges_2[:-1], -hist_2, width=bin_width, color=density_color_2, alpha=0.6)

    ax.plot(
        x_grid_list[window_index_1],
        density_list[window_index_1],
        color=kde_color,
        linewidth=0.5,
    )
    ax.plot(
        x_grid_list[window_index_2],
        -density_list[window_index_2],
        color=kde_color,
        linewidth=0.5,
    )

    ax.scatter(
        x_grid_list[window_index_1][peak_index_list[window_index_1]],
        density_list[window_index_1][peak_index_list[window_index_1]],
        c=np.log(density_list[window_index_1][peak_index_list[window_index_1]]),
        cmap="viridis",
        s=7,
        vmin=-1,
        vmax=3,
        zorder=10,
    )
    ax.scatter(
        x_grid_list[window_index_2][peak_index_list[window_index_2]],
        -density_list[window_index_2][peak_index_list[window_index_2]],
        c=np.log(density_list[window_index_2][peak_index_list[window_index_2]]),
        cmap="viridis",
        s=7,
        vmin=-1,
        vmax=3,
        zorder=10,
    )

    ax.set_xlim([0.04, 0.35])
    ax.set_ylim([-24, 24])
    ax.set_xlabel("$\phi$ (revs)")
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)


def plot_phase_trace(
    ax,
    phase_signal,
    window_index,
    kimograph_result,
    sampling_frequency=SAMPLING_FREQUENCY,
    color="red",
):
    """
    Plot the raw phase trajectory for a given window.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object for the phase trace.
    phase_signal : array_like
        Raw phase trajectory.
    window_index : int
        Window index to visualize.
    kimograph_result : dict
        Result of compute_kimograph.
    sampling_frequency : float, optional
        Sampling frequency in Hz.
    color : str, optional
        Marker color for the trace.
    """
    start_list = kimograph_result["window_start_list"]
    stop_list = kimograph_result["window_stop_list"]

    start = start_list[window_index]
    stop = stop_list[window_index]
    x_values = np.arange(start, stop) / sampling_frequency
    x_values -= x_values[0]

    ax.scatter(x_values, phase_signal[start:stop], color=color, s=0.1)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    axgrid = ax.secondary_yaxis("right")
    axgrid.set_ylabel("$\phi$ (revs)", rotation=-90, labelpad=12)
    ax.grid(True, alpha=0.3)


def annotate_axes(axes):
    """
    Annotate a collection of axes with subplot labels.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
        Axes objects to label.
    """
    for index, ax in enumerate(axes):
        ax.text(
            -0.02,
            1.08,
            chr(97 + index),
            transform=ax.transAxes,
            fontsize=7,
            fontweight="bold",
            fontname="Arial",
            va="top",
            ha="right",
        )


def create_figure6(
    ax_kimograph,
    ax_histogram,
    ax_trace,
    phi_unwrapped,
    name,
    folder,
    window=DEFAULT_WINDOW,
    shift=DEFAULT_SHIFT,
    sampling_frequency=SAMPLING_FREQUENCY,
    window_indices=(58, 167),
):
    """
    Fill existing axes with Figure 6 content.

    Parameters
    ----------
    ax_kimograph : matplotlib.axes.Axes
        Axes for the kimograph plot.
    ax_histogram : matplotlib.axes.Axes
        Axes for the histogram panel.
    ax_trace : matplotlib.axes.Axes
        Axes for the phase trace panel.
    phase_signal : array_like
        Raw phase trajectory used for histogram and trace panels.
    name : str
        Substring identifying the data file.
    folder : str
        Folder containing the data files.
    phi_unwrapped : array_like, optional
        Preloaded unwrapped phase trajectory. If None, it is loaded from files.
    window : int, optional
        Kimograph window size in samples.
    shift : int, optional
        Kimograph shift in samples.
    sampling_frequency : float, optional
        Sampling frequency in Hz.
    window_indices : tuple of int, optional
        Indices of the two windows used for histogram and trace panels.
    """
    if phi_unwrapped is None:
        _, phi_unwrapped, _, _, _ = load_phase_and_state_data(folder, name)
    phi_unwrapped = np.convolve(phi_unwrapped, np.ones(100) / 100, mode="valid")[
        ::100
    ] / (2 * np.pi)
    print(f"Loaded and decimated unwrapped phase with {len(phi_unwrapped)} samples")

    kimograph_result = compute_kimograph(
        phi_unwrapped,
        window=window,
        shift=shift,
        sampling_frequency=sampling_frequency,
        bandwidth=DEFAULT_KDE_BANDWIDTH,
        resolution=DEFAULT_KDE_RESOLUTION,
        prominence=DEFAULT_PROMINENCE,
        min_peak_distance=DEFAULT_STATE_DISTANCE,
    )

    plot_kimograph(
        ax_kimograph,
        phi_unwrapped,
        kimograph_result,
        shift=shift,
        sampling_frequency=sampling_frequency,
    )

    plot_phase_histograms(
        ax_histogram,
        phi_unwrapped,
        kimograph_result,
        window_indices[0],
        window_indices[1],
    )

    plot_phase_trace(
        ax_trace,
        phi_unwrapped,
        window_indices[0],
        kimograph_result,
        sampling_frequency=sampling_frequency,
        color="red",
    )
    plot_phase_trace(
        ax_trace,
        phi_unwrapped,
        window_indices[1],
        kimograph_result,
        sampling_frequency=sampling_frequency,
        color="black",
    )

    annotate_axes([ax_kimograph, ax_histogram, ax_trace])

    return kimograph_result


def build_figure6(
    folder,
    name,
    window=DEFAULT_WINDOW,
    shift=DEFAULT_SHIFT,
    sampling_frequency=SAMPLING_FREQUENCY,
    window_indices=(58, 167),
    figsize=(12, 6),
):
    """
    Build Figure 6 for a specific dataset and return the matplotlib Figure.

    Parameters
    ----------
    folder : str
        Path to the folder containing the data files.
    name : str
        Substring used to identify the dataset files.
    window : int, optional
        Kimograph window size in samples.
    shift : int, optional
        Kimograph window shift in samples.
    sampling_frequency : float, optional
        Sampling frequency in Hz.
    window_indices : tuple of int, optional
        Two window indices used for the histogram and trace panels.
    figsize : tuple, optional
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the Figure 6 panels.
    kimograph_result : dict
        Result dictionary returned by compute_kimograph.
    """
    fig, (ax_kimograph, ax_histogram, ax_trace) = plt.subplots(
        1, 3, figsize=figsize, constrained_layout=True
    )

    xt, phi_unwrapped, _, _, _ = load_phase_and_state_data(folder, name)
    kimograph_result = create_figure6(
        ax_kimograph,
        ax_histogram,
        ax_trace,
        phi_unwrapped,
        name,
        folder,
        window=window,
        shift=shift,
        sampling_frequency=sampling_frequency,
        window_indices=window_indices,
    )
    plt.show()
    return fig, kimograph_result


build_figure6(
    folder="C:\\Users\\rieu\\OneDrive - Nexus365\\PapierRBANMRAC\\Data\\good",
    name="DeltaMotAB_Motor4",
    window=DEFAULT_WINDOW,
    shift=DEFAULT_SHIFT,
    sampling_frequency=SAMPLING_FREQUENCY,
    window_indices=(58, 167),
    figsize=(12, 6),
)
