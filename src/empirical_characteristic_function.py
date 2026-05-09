"""
Empirical Characteristic Functions (ECF) Analysis

This module provides methods for computing empirical characteristic functions (ECF)
from angular trajectories, computing ECF statistics across multiple trajectories,
and analyzing ECF mode properties.

The ECF of a trajectory is defined as:
ECF(n) = |<exp(i*n*phi(t))>| where n is the mode number and <> denotes
time or ensemble average.

Features:
- Distributed computation using Ray for efficient processing
- Local prominence analysis for mode significance assessment
- Multi-trajectory ECF averaging and comparison

Dependencies: numpy, ray, tqdm, matplotlib
"""

import numpy as np
import ray
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================================
# OPTICAL PARAMETERS (capital letters - adjust as needed)
# ============================================================================
SAMPLING_FREQUENCY = 250000  # Hz

# ECF computation parameters
ECF_NSTEPMAX_DEFAULT = 900000  # maximum number of iterations/windows
ECF_NWINDOW_DEFAULT = 100000  # window size in samples
ECF_NSHIFT_DEFAULT_FACTOR = 0.5  # shift factor relative to window size
ECF_MODEMIN_DEFAULT = 1  # minimum mode to study
ECF_MODEMAX_DEFAULT = 60  # maximum mode to study
ECF_NWINDOW_REDUCED = 100  # reduced window for quick analysis
ECF_NSHIFT_REDUCED = 50  # reduced shift for quick analysis

# Local prominence analysis
NUM_ECF_MODES = 59  # Number of modes in standard analysis
LOCAL_PROMINENCE_WINDOW = 3  # Window size for local minimum detection
MIN_PROMINENCE_RATIO = 1.5  # Significance threshold for peak detection


@ray.remote
def compute_ecf_mode_distributed(phi_remote_id, windows_list, mode_number):
    """
    Compute ECF for a single mode (Ray remote function).

    This function is executed remotely on Ray workers. It computes the ECF
    amplitude for a specific mode across multiple time windows.

    Parameters
    ----------
    phi_remote_id : ray.ObjectRef
        Ray object reference to the angular trajectory
    windows_list : list of [int, int]
        List of [start, stop] sample indices for each time window
    mode_number : int
        Mode number for ECF computation

    Returns
    -------
    ecf_magnitude : float
        Sum of ECF magnitudes across all windows for this mode
    mode_number : int
        The mode number (for result identification)
    """
    phi = ray.get(phi_remote_id)
    ecf_magnitude = 0.0

    for window in windows_list:
        start, stop = window[0], window[1]
        # ECF(n) = |sum(exp(i*n*phi))| for samples in this window
        ecf_magnitude += np.abs(np.sum(np.exp(1.0j * phi[start:stop] * mode_number)))

    return ecf_magnitude, mode_number


def ecf_ray_iterator(object_ids):
    """
    Iterator over Ray remote results as they complete.

    Parameters
    ----------
    object_ids : list
        List of Ray object references

    Yields
    ------
    result : tuple
        Result from completed Ray task
    """
    while object_ids:
        done, object_ids = ray.wait(object_ids)
        yield ray.get(done[0])


def compute_empirical_characteristic_function(
    phi_trajectory,
    nstepmax=None,
    nwindow=None,
    start_sample=0,
    nshift=None,
    mode_min=None,
    mode_max=None,
):
    """
    Compute empirical characteristic function (ECF) using distributed processing.

    Calculates the ECF for a trajectory by dividing it into windows and computing
    the characteristic function for each mode using Ray distributed computing.

    Parameters
    ----------
    phi_trajectory : array_like
        Angular trajectory (in radians)
    nstepmax : int, optional
        Maximum number of time windows (default: ECF_NSTEPMAX_DEFAULT)
    nwindow : int, optional
        Size of each window in samples (default: ECF_NWINDOW_DEFAULT)
    start_sample : int, optional
        Starting sample index (default: 0)
    nshift : int, optional
        Step size between consecutive windows (default: nwindow//2)
    mode_min : int, optional
        Minimum mode number (default: ECF_MODEMIN_DEFAULT)
    mode_max : int, optional
        Maximum mode number (default: ECF_MODEMAX_DEFAULT)

    Returns
    -------
    modes : range
        Range of mode numbers analyzed
    ecf_values : ndarray
        ECF amplitude for each mode
    num_windows : int
        Number of time windows used
    windows : list
        List of [start, stop] indices for each window
    """
    # Set defaults
    if nstepmax is None:
        nstepmax = ECF_NSTEPMAX_DEFAULT
    if nwindow is None:
        nwindow = ECF_NWINDOW_DEFAULT
    if nshift is None:
        nshift = int(nwindow * ECF_NSHIFT_DEFAULT_FACTOR)
    if mode_min is None:
        mode_min = ECF_MODEMIN_DEFAULT
    if mode_max is None:
        mode_max = ECF_MODEMAX_DEFAULT

    # Shutdown and reinitialize Ray
    try:
        ray.shutdown()
    except:
        pass
    ray.init()

    # Put trajectory in Ray object store for efficient access
    phi_remote = ray.put(phi_trajectory)

    trajectory_length = len(phi_trajectory)
    mode_list = range(mode_min, mode_max)
    ecf_values = np.zeros(len(mode_list))

    # Calculate number of windows
    num_windows = int((trajectory_length - nwindow - start_sample) / nshift) + 1
    if num_windows > nstepmax:
        num_windows = nstepmax

    # Create window list
    windows = []
    for window_idx in range(num_windows):
        window_start = window_idx * nshift + start_sample
        window_stop = start_sample + window_idx * nshift + nwindow
        windows.append([window_start, window_stop])

    # Submit distributed tasks
    result_ids = [
        compute_ecf_mode_distributed.remote(phi_remote, windows, mode)
        for mode in mode_list
    ]
    results = []

    # Collect results as they complete
    for result in tqdm(
        ecf_ray_iterator(result_ids), total=len(result_ids), desc="Computing ECF modes"
    ):
        results.append(result)

    # Organize results by mode
    for result_idx in range(len(results)):
        mode_number = results[result_idx][1]
        mode_position = np.where(np.array(list(mode_list)) == mode_number)[0]
        ecf_values[mode_position] = results[result_idx][0]

    ray.shutdown()

    return mode_list, ecf_values, num_windows, windows


def generate_ecf_files(unwrapped_phi_list, output_file_names):
    """
    Generate and save ECF files for multiple trajectories.

    Computes ECF for each trajectory and saves results to numpy files.

    Parameters
    ----------
    unwrapped_phi_list : list of ndarray
        List of unwrapped angular trajectories
    output_file_names : list of str
        List of output file paths (without extension)
    """
    for trajectory_idx, (phi_trajectory, file_name) in enumerate(
        zip(unwrapped_phi_list, output_file_names)
    ):
        print(f"Processing trajectory {trajectory_idx + 1}/{len(unwrapped_phi_list)}")

        # Compute ECF with default parameters
        mode_list, ecf_values, num_windows, windows = (
            compute_empirical_characteristic_function(
                phi_trajectory,
                nstepmax=ECF_NSTEPMAX_DEFAULT,
                nwindow=ECF_NWINDOW_DEFAULT,
                nshift=int(ECF_NWINDOW_DEFAULT * ECF_NSHIFT_DEFAULT_FACTOR),
                mode_min=ECF_MODEMIN_DEFAULT,
                mode_max=ECF_MODEMAX_DEFAULT,
            )
        )

        # Save results
        output_path = f"{file_name}_ECF.npy"
        np.save(output_path, np.vstack((list(mode_list), ecf_values)))
        print(f"Saved: {output_path}")


def compute_local_prominence(ecf_values, window_size=None):
    """
    Compute local prominence of ECF peaks.

    Calculates the ratio between each mode's ECF value and the local minimum
    in its neighborhood. High prominence indicates significant modes.

    Parameters
    ----------
    ecf_values : array_like
        ECF amplitudes for each mode
    window_size : int, optional
        Half-width of neighborhood for local minimum (default: LOCAL_PROMINENCE_WINDOW)

    Returns
    -------
    prominence : list
        Local prominence values for each mode
    """
    if window_size is None:
        window_size = LOCAL_PROMINENCE_WINDOW

    prominence = []

    for mode_idx in range(len(ecf_values)):
        # Find local minimum in neighborhood
        start_idx = max(0, mode_idx - window_size)
        end_idx = min(len(ecf_values), mode_idx + 1)

        local_minimum = np.min(ecf_values[start_idx:end_idx])

        if local_minimum > 0:
            peak_prominence = ecf_values[mode_idx] / local_minimum
        else:
            peak_prominence = np.inf if ecf_values[mode_idx] > 0 else 1.0

        prominence.append(peak_prominence)

    return prominence


def load_and_average_ecf_files(folder_pattern, num_files=None):
    """
    Load ECF files and compute average ECF across trajectories.

    Loads ECF data from multiple files, computes mean ECF and mean local
    prominence across all trajectories.

    Parameters
    ----------
    folder_pattern : str
        Pattern or folder path for ECF files
    num_files : int, optional
        Maximum number of files to process

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'modes': Mode numbers
        - 'average_ecf': Average ECF across all trajectories
        - 'average_prominence': Average local prominence
        - 'num_trajectories': Number of trajectories processed
    """
    average_ecf = np.zeros(NUM_ECF_MODES)
    average_prominence = np.zeros(NUM_ECF_MODES)
    trajectory_count = 0

    # Find ECF files (this would need adjustment based on actual file organization)
    import glob

    ecf_files = glob.glob(f"{folder_pattern}*_ECF.npy")

    if num_files:
        ecf_files = ecf_files[:num_files]

    for ecf_file in ecf_files:
        try:
            data = np.load(ecf_file)
            modes = data[0, :NUM_ECF_MODES]
            ecf_amps = data[1, :NUM_ECF_MODES]

            # Accumulate
            average_ecf += ecf_amps
            prominence = compute_local_prominence(ecf_amps)
            average_prominence += np.array(prominence[:NUM_ECF_MODES])

            trajectory_count += 1

        except Exception as e:
            print(f"Warning: Could not process {ecf_file}: {e}")

    # Normalize by number of trajectories
    if trajectory_count > 0:
        average_ecf /= trajectory_count
        average_prominence /= trajectory_count

    return {
        "modes": np.arange(1, NUM_ECF_MODES + 1),
        "average_ecf": average_ecf,
        "average_prominence": average_prominence,
        "num_trajectories": trajectory_count,
    }


def plot_ecf_analysis(average_ecf, average_prominence, modes=None):
    """
    Create publication-quality plots of ECF analysis results.

    Plots average ECF and average prominence across modes with clean formatting.

    Parameters
    ----------
    average_ecf : array_like
        Average ECF amplitudes
    average_prominence : array_like
        Average local prominence values
    modes : array_like, optional
        Mode numbers (default: 1 to len(average_ecf))
    """
    if modes is None:
        modes = np.arange(1, len(average_ecf) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Average ECF
    ax1.plot(
        modes,
        average_ecf,
        color="red",
        linewidth=2,
        marker="o",
        markersize=2,
        label="Average ECF",
    )
    ax1.set_xlabel("Mode number", fontsize=12)
    ax1.set_ylabel("Average ECF (a.u.)", fontsize=12)
    ax1.set_title("Empirical Characteristic Function")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_yticks([])

    # Plot 2: Average Prominence
    ax2.plot(
        modes,
        average_prominence,
        color="red",
        linewidth=2,
        marker="s",
        markersize=2,
        label="Average prominence",
    )
    ax2.set_xlabel("Mode number", fontsize=12)
    ax2.set_ylabel("Average prominence (a.u.)", fontsize=12)
    ax2.set_title("Local Prominence of ECF Peaks")
    ax2.set_ylim([0.9, 1.7])
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_yticks([])

    plt.tight_layout()

    return fig

