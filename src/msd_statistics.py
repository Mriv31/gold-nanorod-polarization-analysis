"""
Mean Squared Displacement (MSD) and Statistical Analysis

This module provides methods for computing time-averaged MSD (TA-MSD) from angular
trajectories, fitting MSD data to various kinetic models, and visualizing results.

Supported models:
- Power law: A * tau^gamma
- Exponential: A * (1 - exp(-tau/t0))
- Stretched exponential: A * (1 - exp(-(tau/t0)^beta))
- Power log: A * ln^gamma(tau/t0)

Dependencies: numpy, scipy.signal, scipy.optimize, matplotlib
"""

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from matplotlib.ticker import NullFormatter
import matplotlib.pyplot as plt


# ============================================================================
# OPTICAL PARAMETERS (capital letters - adjust as needed)
# ============================================================================
SAMPLING_FREQUENCY = 250000  # Hz
MIN_LAG = 4  # samples
MAX_LAG = 5000  # samples
NUM_LAG_POINTS = 200  # logarithmic spacing

# Fitting parameters
MIN_FREQUENCY_SMOOTHING = 200  # Hz - threshold for log smoothing
INITIAL_SMOOTHING_WINDOW = 5
SMOOTHING_FACTOR = 0.01
PERCENTILE_LOW = 5
PERCENTILE_HIGH = 95

# MSD fit bounds
T0_DEFAULT = 4e-6  # seconds
GAMMA_BOUNDS = [0, 2]
LAMBDA_BOUNDS = [0, 1e6]
STRETCHED_GAMMA_BOUNDS = [0, 2]


def calculate_msd_statistics(trajectory, time_lags):
    """
    Compute time-averaged MSD (TA-MSD) statistics for a trajectory.

    Calculates mean, standard deviation, and percentiles of squared displacements
    at specified time lags from an angular trajectory.

    Parameters
    ----------
    trajectory : array_like
        Angular trajectory (in radians, typically unwrapped phase)
    time_lags : array_like
        Time lag indices (in samples) at which to compute MSD

    Returns
    -------
    msd_mean : ndarray
        Mean squared displacement for each lag
    msd_std : ndarray
        Standard deviation of squared displacements
    msd_5th_percentile : ndarray
        5th percentile of squared displacements
    msd_95th_percentile : ndarray
        95th percentile of squared displacements
    sample_count : ndarray
        Number of displacement samples at each lag
    """
    msd_mean = []
    msd_std = []
    msd_5th_percentile = []
    msd_95th_percentile = []
    sample_count = []

    for lag in time_lags:
        # Calculate squared displacements at this lag
        displacements_squared = (trajectory[lag::lag] - trajectory[:-lag:lag]) ** 2

        msd_mean.append(np.mean(displacements_squared))
        msd_std.append(np.std(displacements_squared))
        msd_5th_percentile.append(np.percentile(displacements_squared, PERCENTILE_LOW))
        msd_95th_percentile.append(
            np.percentile(displacements_squared, PERCENTILE_HIGH)
        )
        sample_count.append(len(displacements_squared))

    return (
        np.array(msd_mean),
        np.array(msd_std),
        np.array(msd_5th_percentile),
        np.array(msd_95th_percentile),
        np.array(sample_count),
    )


def plot_msd_statistics_multiple_trajectories(ax, trajectory_files, time_lags):
    """
    Plot MSD statistics from multiple trajectories with error bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    trajectory_files : list of str
        List of file paths containing unwrapped phase trajectories
    time_lags : array_like
        Time lag indices (in samples) at which to compute MSD
    """
    for trajectory_file in trajectory_files:
        try:
            phi_unwrapped = np.load(trajectory_file)
            (
                msd_mean_values,
                msd_std_values,
                msd_5th_percentile_values,
                msd_95th_percentile_values,
                sample_count_values,
            ) = calculate_msd_statistics(phi_unwrapped, time_lags)

            # Convert time lags to seconds
            time_lags_seconds = time_lags[: len(msd_mean_values)] / SAMPLING_FREQUENCY
            standard_error = msd_std_values / np.sqrt(sample_count_values)

            ax.errorbar(
                time_lags_seconds,
                msd_mean_values,
                fmt="o",
                yerr=standard_error,
                elinewidth=0.5,
                markersize=1,
                capsize=0,
                alpha=0.7,
                label=trajectory_file,
            )

        except FileNotFoundError:
            print(f"Warning: File not found - {trajectory_file}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("TA-MSD $\delta(\\tau)$ $(rad^2)$")
    ax.set_xlabel("$\\tau$ (s)", labelpad=0)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)


def smooth_spectrum_logarithmic(
    frequencies, spectrum, initial_window_size=None, smoothing_factor=None
):
    """
    Apply logarithmic smoothing to a spectrum with increasing window size.

    Implements adaptive smoothing where window size increases logarithmically
    for frequencies above a threshold to reduce noise while preserving structure.

    Parameters
    ----------
    frequencies : array_like
        Frequency values
    spectrum : array_like
        Spectral values to smooth
    initial_window_size : int, optional
        Initial window size (default: INITIAL_SMOOTHING_WINDOW)
    smoothing_factor : float, optional
        Factor controlling window growth rate (default: SMOOTHING_FACTOR)

    Returns
    -------
    smoothed_spectrum : ndarray
        Smoothed spectrum values
    """
    if initial_window_size is None:
        initial_window_size = INITIAL_SMOOTHING_WINDOW
    if smoothing_factor is None:
        smoothing_factor = SMOOTHING_FACTOR

    smoothed_spectrum = np.copy(spectrum)
    window_size = initial_window_size
    window_size_raw = float(window_size)

    for i in range(len(frequencies)):
        if frequencies[i] > MIN_FREQUENCY_SMOOTHING:
            start = max(0, i - window_size // 2)
            end = min(len(frequencies), i + window_size // 2 + 1)
            smoothed_spectrum[i] = np.mean(spectrum[start:end])
            window_size_raw += smoothing_factor
            window_size = int(window_size_raw)

    return smoothed_spectrum


def plot_power_spectral_density(ax, phi_unwrapped):
    """
    Plot power spectral density with Welch method.

    Computes and displays the PSD of an angular trajectory with logarithmic
    frequency axis and minimal tick marks for publication-quality figures.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    phi_unwrapped : array_like
        Unwrapped angular trajectory (in radians)
    """
    # Compute PSD using Welch's method
    frequencies, power_spectral_density = signal.welch(
        phi_unwrapped, fs=SAMPLING_FREQUENCY, nperseg=25000
    )

    ax.semilogy(frequencies, power_spectral_density)
    ax.set_xlabel("f (Hz)", labelpad=0)
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000, 10000, 100000])
    ax.tick_params(axis="x", pad=0)
    ax.set_ylabel("PSD $(rad^2/Hz)$", labelpad=0)

    # Add secondary axis for dual labeling
    secondary_ax = ax.secondary_yaxis("right")
    secondary_ax.tick_params(axis="y", which="both", pad=0)

    # Clean up y-axis formatting
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_tick_params(which="both", length=0)


def power_law_model(time_lag, gamma, amplitude):
    """Power law model: A * tau^gamma"""
    return amplitude * time_lag**gamma


def exponential_model(time_lag, lambda_param, amplitude):
    """Exponential model: A * (1 - exp(-tau/t0))"""
    return amplitude * (1 - np.exp(-lambda_param * time_lag))


def stretched_exponential_model(time_lag, lambda_param, stretch_exponent, amplitude):
    """Stretched exponential model: A * (1 - exp(-(tau/t0)^beta))"""
    return amplitude * (1 - np.exp(-((lambda_param * time_lag) ** stretch_exponent)))


def power_log_model(time_lag, t0, gamma, amplitude):
    """Power log model: A * ln^gamma(tau/t0)"""
    return amplitude * np.log(time_lag / t0) ** gamma


def fit_msd_to_models(time_lags, msd_mean, msd_std, t0=None):
    """
    Fit MSD data to multiple kinetic models.

    Attempts to fit the MSD curve to exponential, stretched exponential, and
    power-law models using weighted least squares.

    Parameters
    ----------
    time_lags : array_like
        Time lag indices (in samples)
    msd_mean : array_like
        Mean squared displacement values
    msd_std : array_like
        Standard deviation of MSD (used as weights)
    t0 : float, optional
        Reference time for power-law model (default: T0_DEFAULT)

    Returns
    -------
    fits : dict
        Dictionary with fitted parameters for each model:
        - 'exponential': (lambda, amplitude, covariance)
        - 'stretched_exponential': (lambda, beta, amplitude, covariance)
        - 'power_log': (t0, gamma, amplitude, covariance)
    """
    if t0 is None:
        t0 = T0_DEFAULT

    fits = {}

    # Filter out zero or invalid values
    valid_idx = (msd_mean > 0) & (msd_std > 0)
    time_lags_valid = time_lags[valid_idx]
    msd_mean_valid = msd_mean[valid_idx]
    msd_std_valid = msd_std[valid_idx]

    # Exponential fit
    try:
        popt_exp, pcov_exp = curve_fit(
            exponential_model,
            time_lags_valid,
            msd_mean_valid,
            sigma=msd_std_valid,
            p0=[1e5, msd_mean_valid[-1]],
            maxfev=10000,
        )
        fits["exponential"] = (popt_exp, pcov_exp)
    except RuntimeError:
        print("Exponential fit failed")
        fits["exponential"] = (None, None)

    # Stretched exponential fit
    try:
        popt_stretched, pcov_stretched = curve_fit(
            stretched_exponential_model,
            time_lags_valid,
            msd_mean_valid,
            p0=[1e4, 0.2, msd_mean_valid[-1]],
            sigma=msd_std_valid,
            maxfev=10000,
        )
        fits["stretched_exponential"] = (popt_stretched, pcov_stretched)
    except RuntimeError:
        print("Stretched exponential fit failed")
        fits["stretched_exponential"] = (None, None)

    # Power log fit
    try:
        # Initial log-log fit for power-law estimate
        log_time = np.log(time_lags_valid / t0)
        log_msd = np.log(msd_mean_valid)
        a_log, b_log = np.polyfit(log_time, log_msd, 1)

        popt_power_log, pcov_power_log = curve_fit(
            power_log_model,
            time_lags_valid,
            msd_mean_valid,
            p0=[t0, a_log, np.exp(b_log)],
            bounds=([1e-10, 0, 0], [1e-3, 2, np.inf]),
            sigma=msd_std_valid,
            maxfev=10000,
        )
        fits["power_log"] = (popt_power_log, pcov_power_log)
    except RuntimeError:
        print("Power-log fit failed")
        fits["power_log"] = (None, None)

    return fits


def plot_msd_with_fits(ax, time_lags, msd_mean, msd_std, fits=None):
    """
    Plot MSD data with fitted models.

    Creates a log-log plot of MSD with error bars and optional fitted curves
    for exponential, stretched exponential, and power-law models.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    time_lags : array_like
        Time lag indices (in samples)
    msd_mean : array_like
        Mean squared displacement values
    msd_std : array_like
        Standard deviation of MSD
    fits : dict, optional
        Dictionary of fitted parameters from fit_msd_to_models
    """
    time_lags_seconds = time_lags / SAMPLING_FREQUENCY

    ax.errorbar(
        time_lags_seconds,
        msd_mean,
        fmt=".",
        markersize=1,
        yerr=msd_std,
        linewidth=0.5,
        label="Data",
    )

    if fits is not None:
        # Plot fitted curves
        time_range = np.logspace(
            np.log10(time_lags_seconds.min()), np.log10(time_lags_seconds.max()), 500
        )

        if fits["exponential"][0] is not None:
            popt = fits["exponential"][0]
            ax.plot(
                time_range,
                exponential_model(time_range, *popt),
                linestyle="-",
                linewidth=1,
                label="$A(1-e^{-\\tau/t_0})$",
                color="red",
                zorder=15,
            )

        if fits["stretched_exponential"][0] is not None:
            popt = fits["stretched_exponential"][0]
            ax.plot(
                time_range,
                stretched_exponential_model(time_range, *popt),
                linestyle="-",
                linewidth=1,
                label="$A(1-e^{-(\\tau/t_0)^{\\beta}})$",
                zorder=16,
            )

        if fits["power_log"][0] is not None:
            popt = fits["power_log"][0]
            ax.plot(
                time_range,
                power_log_model(time_range, *popt),
                linestyle="-",
                linewidth=1,
                label="$A \\ln^{\\gamma}(\\tau/t_0)$",
                zorder=17,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$\\tau$ (s)", labelpad=0)
    ax.set_ylabel("TA-MSD $\delta(\\tau)$ $(rad^2)$")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # Add secondary axis
    secondary_ax = ax.secondary_yaxis("right")
    secondary_ax.set_ylabel("TA-MSD $(rad^2)$", labelpad=10, rotation=270)
    secondary_ax.yaxis.set_minor_formatter(NullFormatter())

    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_tick_params(which="both", length=0)
