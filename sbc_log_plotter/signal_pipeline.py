# Author  : jojaebeom@kimm.re.kr
# Modify  : 2026-06-26
"""Stateless signal processing pipeline for SBC log plot.

Pure-function filter and FFT helpers. No Qt / no GUI deps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import butter, get_window, sosfilt, sosfiltfilt, welch


FilterKind = Literal["none", "ma", "lpf", "hpf"]
FilterPhase = Literal["zero", "causal"]
FFTWindow = Literal["hann", "hamming", "rect"]
FFTScale = Literal["linear", "db"]
FFTDetrend = Literal["none", "mean"]


@dataclass(frozen=True)
class FilterCfg:
    kind: FilterKind = "none"
    window: int = 5
    fc: float = 20.0
    order: int = 4
    phase: FilterPhase = "zero"


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or arr.size <= 1:
        return arr.copy()
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    idx = np.arange(arr.size)
    start = np.maximum(0, idx - window + 1)
    counts = (idx - start + 1).astype(float)
    return (cumsum[idx + 1] - cumsum[start]) / counts


def _butter_sos(fs: float, cfg: FilterCfg) -> np.ndarray:
    btype = "low" if cfg.kind == "lpf" else "high"
    return butter(cfg.order, cfg.fc, btype=btype, fs=fs, output="sos")


def _filtfilt_min_len(order: int) -> int:
    # scipy.signal.sosfiltfilt default padlen ≈ (order+1)*3
    return (order + 1) * 3


def apply_filter(values: np.ndarray, fs: float, cfg: FilterCfg) -> np.ndarray:
    """Apply filter; returns a new ndarray. Never raises on edge cases."""
    arr = np.asarray(values, dtype=float)
    if cfg.kind == "none" or arr.size < 2:
        return arr.copy()
    if cfg.kind == "ma":
        return _moving_average(arr, cfg.window)
    if cfg.kind in ("lpf", "hpf"):
        nyq = fs / 2.0
        if cfg.fc <= 0.0 or cfg.fc >= nyq:
            return arr.copy()
        sos = _butter_sos(fs, cfg)
        if cfg.phase == "zero" and arr.size > _filtfilt_min_len(cfg.order):
            return sosfiltfilt(sos, arr)
        return sosfilt(sos, arr)
    raise ValueError(f"Unknown filter kind: {cfg.kind}")


@dataclass(frozen=True)
class FFTCfg:
    window: FFTWindow = "hann"
    scale: FFTScale = "db"
    detrend: FFTDetrend = "mean"
    psd: bool = True
    nperseg: int = 2048


_FFT_WINDOW_NAMES: dict[str, str] = {
    "hann": "hann",
    "hamming": "hamming",
    "rect": "boxcar",
}


def compute_spectrum(
    values: np.ndarray, fs: float, cfg: FFTCfg
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs[Hz], magnitudes). Empty arrays if signal too short."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 8:
        return np.array([]), np.array([])

    window_name = _FFT_WINDOW_NAMES[cfg.window]
    detrend_arg = "constant" if cfg.detrend == "mean" else False

    if cfg.psd:
        nperseg = min(cfg.nperseg, arr.size)
        freqs, pxx = welch(
            arr, fs=fs, window=window_name, nperseg=nperseg,
            detrend=detrend_arg, scaling="density",
        )
        mag = 10.0 * np.log10(pxx + 1e-20) if cfg.scale == "db" else pxx
        return freqs, mag

    x = arr - arr.mean() if cfg.detrend == "mean" else arr
    win = get_window(window_name, x.size)
    spectrum = np.fft.rfft(x * win) / max(np.sum(win), 1e-20)
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    amp = np.abs(spectrum)
    mag = 20.0 * np.log10(amp + 1e-20) if cfg.scale == "db" else amp
    return freqs, mag
