"""
APD Correlation and Phase Stability Analysis

Corresponds to Supplementary Figure S3 in the paper.

Reusable analysis module for studying APD correlation channels and phase stability.
Accepts phi_unwrapped traces and four APD correlation channels to compute histogram peaks,
power spectral density, and Allan deviation.

"""

from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

DEFAULT_FS = 250e3
DEFAULT_DOWNSAMPLE = 1000
DEFAULT_PEAK_HEIGHT = 5e5
DEFAULT_MAX_CHANNEL_VALUE = 5.0
DEFAULT_THRESHOLD_RATIO = 0.1
DEFAULT_MIN_DURATION_S = 5.0
DEFAULT_NPERSEG = int(1e5)
DEFAULT_HIST_BINS = 100


def largest_contiguous_slice(indices: np.ndarray) -> np.ndarray:
    """Return the longest contiguous subsequence of integer indices."""
    if indices.size == 0:
        return indices

    diffs = np.diff(indices)
    breaks = np.nonzero(diffs != 1)[0]
    best_slice = indices[:1]
    start = 0

    for br in np.concatenate([breaks, [len(diffs)]]):
        candidate = indices[start : br + 1]
        if candidate.size > best_slice.size:
            best_slice = candidate
        start = br + 1

    return best_slice


def compute_allan_deviation(
    phi_segment: np.ndarray,
    fs: float = DEFAULT_FS,
    n_tau: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute overlapping Allan deviation from a phase segment."""
    if phi_segment.size < 2:
        return np.array([]), np.array([])

    tau0 = 1.0 / fs
    y = np.diff(phi_segment) / (2.0 * np.pi * tau0)
    max_m = max(1, len(y) // 10)
    m_values = np.unique(np.logspace(0, np.log10(max_m), n_tau).astype(int))

    taus: List[float] = []
    allan_dev: List[float] = []

    for m in m_values:
        if 2 * m >= len(y):
            break

        y_avg = np.convolve(y, np.ones(m, dtype=float) / m, mode="valid")
        diff = y_avg[m:] - y_avg[:-m]
        avar = 0.5 * np.mean(diff**2)
        taus.append(m * tau0)
        allan_dev.append(np.sqrt(avar))

    return np.array(taus), np.array(allan_dev)


def analyze_sup_figure_s3(
    phi_unwrapped: np.ndarray,
    cor_channels: Sequence[np.ndarray],
    fs: float = DEFAULT_FS,
    downsample: int = DEFAULT_DOWNSAMPLE,
    peak_height: float = DEFAULT_PEAK_HEIGHT,
    max_channel_value: float = DEFAULT_MAX_CHANNEL_VALUE,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    nperseg: int = DEFAULT_NPERSEG,
    hist_bins: int = DEFAULT_HIST_BINS,
    plot: bool = False,
) -> Dict[str, Any]:
    """Analyze phi_unwrapped and four APD correlation channels for S3."""
    cor_channels = tuple(np.asarray(ch, dtype=float) for ch in cor_channels)
    if len(cor_channels) != 4:
        raise ValueError("cor_channels must contain exactly four channel arrays.")

    if any(ch.shape != phi_unwrapped.shape for ch in cor_channels):
        raise ValueError(
            "All correlation channels must have the same length as phi_unwrapped."
        )

    ctot = np.sum(np.stack(cor_channels, axis=0), axis=0)
    ctot_ds = ctot[::downsample]
    c_ds = [ch[::downsample] for ch in cor_channels]
    sampling_rate_ds = fs / downsample
    min_samples = int(np.ceil(min_duration_s * sampling_rate_ds))

    counts, bin_edges = np.histogram(ctot_ds, bins=hist_bins)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    peaks, _ = signal.find_peaks(counts, height=peak_height)

    segment_windows: List[Tuple[int, int]] = []
    voltage_means: List[float] = []
    phase_stds: List[float] = []
    phase_means: List[float] = []
    psd_results: List[Dict[str, np.ndarray]] = []
    allan_results: List[Dict[str, np.ndarray]] = []

    figs = {}
    if plot:
        fig_hist, ax_hist = plt.subplots()
        ax_hist.plot(centers, counts, drawstyle="steps-mid")
        ax_hist.plot(centers[peaks], counts[peaks], "ro")
        ax_hist.set_xlabel("Sum of APD voltages")
        ax_hist.set_ylabel("Histogram count")
        ax_hist.set_title("CTOT histogram and detected peaks")

        fig_segments = plt.figure(figsize=(90 / 25.4, 60 / 25.4), dpi=300)
        fig_psd = plt.figure(figsize=(90 / 25.4, 60 / 25.4), dpi=300)
        fig_allan = plt.figure(figsize=(90 / 25.4, 60 / 25.4), dpi=300)
        ax_segment = fig_segments.gca()
        ax_psd = fig_psd.gca()
        ax_allan = fig_allan.gca()
        figs = {
            "hist": fig_hist,
            "segments": fig_segments,
            "psd": fig_psd,
            "allan": fig_allan,
        }

    for peak_idx in peaks:
        target = centers[peak_idx]
        mask = np.abs(ctot_ds - target) < threshold_ratio * np.abs(target)
        for ch in c_ds:
            mask &= ch < max_channel_value

        if not np.any(mask):
            continue

        indices = np.nonzero(mask)[0]
        largest_slice = largest_contiguous_slice(indices)
        if largest_slice.size < min_samples:
            continue
        if largest_slice.size <= 1000:
            continue

        start = int(largest_slice[500] * downsample)
        stop = int(largest_slice[-500] * downsample)
        if stop <= start:
            continue

        segment_windows.append((start, stop))
        voltage_means.append(np.mean(ctot_ds[largest_slice][500:-500]))
        phase_means.append(np.mean(phi_unwrapped[start:stop]))

        frequencies, psd = signal.welch(phi_unwrapped[start:stop], fs, nperseg=nperseg)
        phase_stds.append(np.sqrt(np.trapz(psd, frequencies)))
        psd_results.append({"frequencies": frequencies, "psd": psd})

        taus, allan_dev = compute_allan_deviation(phi_unwrapped[start:stop], fs)
        allan_results.append({"taus": taus, "allan_dev": allan_dev})

        if plot:
            ax_segment.plot(ctot_ds[largest_slice][500:-500], label=f"{target:.2f} V")
            ax_psd.loglog(
                frequencies,
                psd,
                label=f"{np.mean(ctot_ds[largest_slice][500:-500]):.2f} V",
            )
            ax_allan.loglog(
                taus,
                allan_dev,
                "o-",
                label=f"{np.mean(ctot_ds[largest_slice][500:-500]):.2f} V",
            )

    if plot:
        ax_segment.set_xlabel("Downsampled sample index")
        ax_segment.set_ylabel("CTOT (downsampled)")
        ax_segment.legend()
        ax_segment.grid(True)

        ax_psd.set_xlabel("Frequency [Hz]")
        ax_psd.set_ylabel(r"PSD $[rad^2/Hz]$")
        ax_psd.grid(True, which="both")
        ax_psd.legend()

        ax_allan.set_xlabel(r"Averaging time $\tau$ [s]")
        ax_allan.set_ylabel("Allan deviation")
        ax_allan.grid(True, which="both")
        ax_allan.legend()

    result = {
        "segment_windows": segment_windows,
        "voltage_means": np.array(voltage_means),
        "phase_stds": np.array(phase_stds),
        "phase_means": np.array(phase_means),
        "psd_results": psd_results,
        "allan_results": allan_results,
        "histogram": {"counts": counts, "centers": centers},
    }
    if plot:
        result["figures"] = figs
    return result


__all__ = [
    "largest_contiguous_slice",
    "compute_allan_deviation",
    "analyze_sup_figure_s3",
]
