"""Metrics for clock-synchronization experiments."""

from __future__ import annotations

import numpy as np

from src.satellite import Satellite


def healthy_offsets_ns(satellites: list[Satellite]) -> np.ndarray:
    """Return clock offsets for non-faulty satellites."""

    return np.array([sat.offset_ns for sat in satellites if sat.is_healthy], dtype=float)


def max_time_difference_ns(satellites: list[Satellite]) -> float:
    """Peak-to-peak clock difference among healthy nodes."""

    offsets = healthy_offsets_ns(satellites)
    if offsets.size == 0:
        return float("nan")
    return float(np.max(offsets) - np.min(offsets))


def rms_error_ns(satellites: list[Satellite]) -> float:
    """Root-mean-square offset after removing the healthy-node mean."""

    offsets = healthy_offsets_ns(satellites)
    if offsets.size == 0:
        return float("nan")
    centered = offsets - np.mean(offsets)
    return float(np.sqrt(np.mean(centered**2)))


def time_to_convergence(
    times_s: list[float], errors_ns: list[float], threshold_ns: float
) -> float | None:
    """Return the first time at which the error stays below a threshold."""

    for idx, error in enumerate(errors_ns):
        if error <= threshold_ns and all(e <= threshold_ns for e in errors_ns[idx:]):
            return times_s[idx]
    return None
