"""
Transition and Lifetimes Statistics Analysis

This module analyzes transition times between global states and computes lifetimes
from step boundaries detected in time series data. It provides statistical analysis
of state transitions and visualization capabilities for transition time distributions.

The analysis workflow:
1. Compute transition times between well-defined global states
2. Calculate lifetimes from step boundaries
3. Visualize transition time statistics with confidence intervals

Dependencies: numpy, scipy.stats, matplotlib
"""

import numpy as np
from scipy.stats import bootstrap
import matplotlib.pyplot as plt


def compute_transition_times(
    global_states, mapped_states, step_boundaries, sampling_freq
):
    """
    Compute transition times between global states for well-defined state mappings.

    This function calculates the time spent transitioning between adjacent global states
    when local states have been successfully mapped to well-defined global states.

    Parameters
    ----------
    global_states : array_like
        Global state angles (in radians) identified via KDE analysis
    mapped_states : array_like
        Local states mapped to global states (in radians)
    step_boundaries : array_like
        Time indices of step boundaries from step detection
    sampling_freq : float
        Sampling frequency in Hz

    Returns
    -------
    transition_times : ndarray
        2D array of lists containing transition times between state pairs
        Shape: (num_states, num_states), dtype=object
    """
    # Map each local state to the closest global state
    state_indices = np.array(
        [
            np.argmin(np.abs(global_states - mapped_state % (2 * np.pi)))
            for mapped_state in mapped_states
        ],
        dtype=int,
    )

    num_states = len(np.unique(state_indices))
    transition_times = np.empty([num_states, num_states], dtype=object)

    # Initialize transition time storage
    for i in range(num_states):
        for j in range(num_states):
            transition_times[i, j] = []

    # Track current transition durations
    current_transition_duration = np.zeros([num_states, num_states])

    # Calculate transition times for consecutive states
    for i, current_state in enumerate(state_indices[:-1]):
        # Accumulate time in potential transition states (adjacent states)
        for adjacent_state in [
            (current_state - 1) % num_states,
            (current_state + 1) % num_states,
        ]:
            time_increment = (
                step_boundaries[i + 1] - step_boundaries[i]
            ) / sampling_freq
            current_transition_duration[current_state, adjacent_state] += time_increment

        # Record completed transition
        next_state = state_indices[i + 1]
        if current_transition_duration[current_state, next_state] > 0:
            transition_times[current_state, next_state].append(
                current_transition_duration[current_state, next_state]
            )

        # Reset transition timer for this state pair
        current_transition_duration[current_state, next_state] = 0

    return transition_times


def compute_lifetimes(motor_names, step_boundaries_list, sampling_freq=250000):
    """
    Compute lifetimes (dwell times) for each motor from step boundaries.

    Calculates mean lifetimes with bootstrap confidence intervals for each motor's
    step boundaries.

    Parameters
    ----------
    motor_names : list of str
        Names/identifiers for each motor
    step_boundaries_list : list of array_like
        List of step boundary arrays for each motor
    sampling_freq : float, default=250000
        Sampling frequency in Hz

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'means': Mean lifetimes for each motor
        - 'ci_low': Lower confidence interval bounds
        - 'ci_high': Upper confidence interval bounds
        - 'all_lifetimes': Combined lifetimes across all motors
    """
    means = []
    ci_low = []
    ci_high = []
    all_lifetimes = []

    for motor_name, step_boundaries in zip(motor_names, step_boundaries_list):
        # Calculate lifetimes as time differences between steps
        lifetimes = np.diff(step_boundaries) / sampling_freq
        all_lifetimes.extend(lifetimes)

        # Bootstrap confidence interval for mean lifetime
        if len(lifetimes) > 1:
            bs_result = bootstrap((lifetimes,), np.mean, confidence_level=0.95)
            mean_lifetime = np.mean(bs_result.bootstrap_distribution)
            ci_lower = bs_result.confidence_interval.low
            ci_upper = bs_result.confidence_interval.high
        else:
            # Handle single lifetime case
            mean_lifetime = lifetimes[0] if len(lifetimes) > 0 else 0
            ci_lower = mean_lifetime
            ci_upper = mean_lifetime

        means.append(mean_lifetime)
        ci_low.append(ci_lower)
        ci_high.append(ci_upper)

        print(
            f"{motor_name}: mean={mean_lifetime:.6f}, "
            f"CI=[{ci_lower:.6f}, {ci_upper:.6f}]"
        )

    return {
        "means": np.array(means),
        "ci_low": np.array(ci_low),
        "ci_high": np.array(ci_high),
        "all_lifetimes": np.array(all_lifetimes),
    }


def plot_transition_times(ax, transition_times):
    """
    Plot transition times between states with error bars and sample counts.

    Creates a log-scale plot showing transition times between adjacent states
    with bootstrap confidence intervals.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    transition_times : ndarray
        2D array of transition time lists from compute_transition_times
    """
    num_states = transition_times.shape[0]
    state_indices = np.arange(num_states)

    # Storage for statistics
    mean_times = np.zeros([num_states, num_states])
    ci_upper = np.zeros([num_states, num_states])
    ci_lower = np.zeros([num_states, num_states])

    min_time, max_time = 1e10, -1

    # Calculate statistics for each state transition
    for i in range(num_states):
        for j_idx, j in enumerate([(i - 1) % num_states, (i + 1) % num_states]):
            times = np.array(transition_times[i, j])

            if len(times) == 0:
                continue

            # Calculate bootstrap statistics
            if len(times) > 1:
                bs_result = bootstrap((times,), np.mean, confidence_level=0.95)
                mean_times[i, j] = np.mean(bs_result.bootstrap_distribution)
                ci_upper[i, j] = bs_result.confidence_interval.high
                ci_lower[i, j] = bs_result.confidence_interval.low
            else:
                # Single sample case
                mean_times[i, j] = times[0]
                ci_upper[i, j] = times[0]
                ci_lower[i, j] = times[0]

            # Choose color and marker based on transition direction
            color = "blue" if j_idx == 0 else "red"
            marker = "<" if j_idx == 0 else ">"

            # Plot with error bars
            ax.errorbar(
                [i],
                [mean_times[i, j]],
                yerr=[
                    [mean_times[i, j] - ci_lower[i, j]],
                    [ci_upper[i, j] - mean_times[i, j]],
                ],
                color=color,
                marker=marker,
                markersize=3,
                capsize=2,
            )

            # Track min/max for axis scaling
            min_time = min(min_time, mean_times[i, j])
            max_time = max(max_time, mean_times[i, j])

        # Add sample count annotations
        counterclockwise_count = len(transition_times[i, (i - 1) % num_states])
        clockwise_count = len(transition_times[i, (i + 1) % num_states])

        ax.text(
            i - 0.5,
            6e-4,
            f"n={counterclockwise_count}",
            rotation=90,
            fontsize=5,
            color="blue",
        )
        ax.text(
            i - 0.5, 1.5, f"n={clockwise_count}", rotation=90, fontsize=5, color="red"
        )

    # Configure axes
    ax.set_xticks(state_indices)
    ax.set_xticklabels(1 + state_indices, rotation=90)
    ax.set_xlabel("State", labelpad=5)
    ax.set_ylabel("Transition time (s)")
    ax.set_yscale("log")
    ax.set_ylim([4e-4, 13])
    ax.tick_params(axis="x", pad=0)

    print(f"Transition time range: {min_time:.2e} to {max_time:.2e} seconds")


def analyze_state_transitions(
    global_states_file, mapped_states_file, step_boundaries_file, sampling_freq=250000
):
    """
    Complete analysis pipeline for state transitions and lifetimes.

    Parameters
    ----------
    global_states_file : str
        Path to numpy file containing global states
    mapped_states_file : str
        Path to numpy file containing mapped states
    step_boundaries_file : str
        Path to numpy file containing step boundaries
    sampling_freq : float, default=250000
        Sampling frequency in Hz

    Returns
    -------
    results : dict
        Analysis results including transition times and statistics
    """
    # Load data
    global_states = np.load(global_states_file)
    mapped_states = np.load(mapped_states_file)
    step_boundaries = np.load(step_boundaries_file)

    # Compute transition times
    transition_times = compute_transition_times(
        global_states, mapped_states, step_boundaries, sampling_freq
    )

    return {
        "transition_times": transition_times,
        "global_states": global_states,
        "mapped_states": mapped_states,
        "step_boundaries": step_boundaries,
    }


if __name__ == "__main__":
    """
    Example usage of transition and lifetime analysis.

    This example demonstrates:
    1. Loading data from previous analysis steps
    2. Computing transition times between global states
    3. Visualizing transition time statistics
    """

    # File paths (adjust as needed for your data)
    global_states_file = "global_states.npy"
    mapped_states_file = "mapped_levels.npy"
    step_boundaries_file = "filtered_boundaries.npy"

    try:
        # Run complete analysis
        results = analyze_state_transitions(
            global_states_file, mapped_states_file, step_boundaries_file
        )

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_transition_times(ax, results["transition_times"])

        plt.tight_layout()
        plt.savefig("transition_times_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("Analysis complete. Results saved to transition_times_analysis.png")

    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Please ensure the following files exist:")
        print("- global_states.npy (from map_local_to_global_states.py)")
        print("- mapped_levels.npy (from map_local_to_global_states.py)")
        print("- filtered_boundaries.npy (from step_detection.py)")
    except Exception as e:
        print(f"Analysis failed: {e}")
