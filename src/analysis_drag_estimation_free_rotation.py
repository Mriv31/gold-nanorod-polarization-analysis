"""
Drag Estimation for Freely Rotating Nanorods

Corresponds to Supplementary Figure S7 in the paper.

This module analyzes freely rotating gold nanorod data using precomputed orientation traces.
It accepts precomputed phi_unwrapped traces  for each
trace, estimating drag coefficients from Welch power spectral density fits.
"""

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

DRAG_CONSTANT = 2.0 / np.pi**2
DEFAULT_FS = 250e3
DEFAULT_NPERSEG = 2**17
DEFAULT_NOVERLAP = 2**16
DEFAULT_FIT_START_IDX = 100


def estimate_drag_from_phi(
    phi_unwrapped: np.ndarray,
    fs: float = DEFAULT_FS,
    nperseg: int = DEFAULT_NPERSEG,
    noverlap: int = DEFAULT_NOVERLAP,
    fit_start_idx: int = DEFAULT_FIT_START_IDX,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Estimate drag from a phi unwrapped trace using Welch PSD analysis."""
    f0, Pxx_den0 = signal.welch(phi_unwrapped, fs, nperseg=nperseg, noverlap=noverlap)
    fit_x = np.log(f0[fit_start_idx:])
    fit_y = 2 * np.log(f0[fit_start_idx:]) + np.log(Pxx_den0[fit_start_idx:])
    pm, pcov = np.polyfit(fit_x, fit_y, 0, cov=True)
    b = pm[0]
    err = np.sqrt(pcov[0][0])

    drag = DRAG_CONSTANT * np.exp(-b)
    drag_error = abs(DRAG_CONSTANT * np.exp(-(b + err)) - drag)
    return f0, Pxx_den0, drag, drag_error, b


def create_sup_figure_s7(
    phi_unwrapped_list: Sequence[np.ndarray],
    tet_cor_channel_list: Optional[
        Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ] = None,
    labels: Optional[Sequence[str]] = None,
    fs: float = DEFAULT_FS,
) -> Tuple[plt.Figure, plt.Figure, np.ndarray, np.ndarray]:
    """Create Figure S7 panels from a list of unwrapped phi traces.

    Parameters
    ----------
    phi_unwrapped_list : Sequence[np.ndarray]
        List of unwrapped azimuthal angle traces.
    tet_cor_channel_list : optional
        Optional list of four correlation channel arrays for each trace.
    labels : optional
        Labels for each trace. If not provided, letters A, B, ... are used.
    fs : float, optional
        Sampling frequency in Hz.

    Returns
    -------
    fig : plt.Figure
        Figure containing individual PSD panels.
    fig2 : plt.Figure
        Figure containing the filtered PSD overlay.
    drag_values : np.ndarray
        Estimated drag coefficients for each trace.
    drag_errors : np.ndarray
        Estimated drag uncertainties for each trace.
    """
    if labels is None:
        labels = [chr(65 + i) for i in range(len(phi_unwrapped_list))]

    fig = plt.figure(figsize=(180 / 25.4, 180 / 25.4), dpi=200)
    gs = fig.add_gridspec(4, 4)
    fig2 = plt.figure(dpi=200)
    ax256 = fig2.add_subplot(1, 1, 1)

    drag_values = []
    drag_errors = []

    subplot_positions = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (3, 0),
    ]

    for i, phi_unwrapped in enumerate(phi_unwrapped_list):
        if i >= len(subplot_positions):
            break

        row, col = subplot_positions[i]
        ax = fig.add_subplot(gs[row, col])
        ax.text(
            -0.14,
            1.1,
            labels[i],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )

        f0, Pxx_den0, drag, drag_error, b = estimate_drag_from_phi(phi_unwrapped, fs=fs)
        drag_values.append(drag)
        drag_errors.append(drag_error)

        filtered_pxx_den0 = np.convolve(Pxx_den0 * f0**2, np.ones(50) / 50, mode="same")
        ax256.plot(f0, filtered_pxx_den0, label=labels[i])

        ax.plot(f0, Pxx_den0, color="black")
        ax.plot(
            f0,
            np.exp(-2 * np.log(f0) + b),
            color="red",
            label=r"$PSD = b f^{-2}$"
            + "\n"
            + r"$\gamma = \frac{kT}{2 b \pi^2}$"
            + f" = {drag:.1e}",
        )
        ax.set_xlabel("Freq (Hz)")
        ax.set_ylabel(r"PSD (rad$^2$/Hz)")
        ax.set_yticks([1e-7, 1e-5, 1e-3, 1e-1, 1e1])
        ax.set_xticks([1e1, 1e2, 1e3, 1e4])
        ax.legend()
        ax.set_yscale("log")
        ax.set_xscale("log")

        if i != 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        if i != len(phi_unwrapped_list) - 1:
            if row != 3 or col != 0:
                ax.set_xticklabels([])
                ax.set_xlabel("")

    summary_ax = fig.add_subplot(gs[2:, 1:])
    summary_ax.errorbar(
        np.arange(len(drag_values)),
        drag_values,
        yerr=drag_errors,
        fmt=".",
        color="black",
    )
    summary_ax.text(
        -0.04,
        1.02,
        "K",
        transform=summary_ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

    summary_ax.set_xticks(np.arange(len(drag_values)))
    summary_ax.set_xticklabels(labels[: len(drag_values)])
    summary_ax.set_xlabel("Rod label")
    summary_ax.set_yscale("log")
    summary_ax.set_yticks([])
    summary_ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    ax2 = summary_ax.secondary_yaxis("right")
    ax2.set_yscale("log")
    ax2.set_ylabel("Drag γ (pN.nm.s.rad$^{-1}$)", rotation=270, labelpad=15)

    ax256.set_xscale("log")
    ax256.set_yscale("log")

    return fig, fig2, np.array(drag_values), np.array(drag_errors)
