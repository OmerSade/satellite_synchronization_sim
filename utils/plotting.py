"""Plotting helpers for synchronization experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_error_comparison(
    times_s: list[float],
    isdts_errors_ns: list[float],
    ptp_errors_ns: list[float],
    output_path: str | Path = "outputs/error_comparison.png",
) -> Path:
    """Plot peak-to-peak time error for IS-DTS and the PTP baseline."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(times_s, isdts_errors_ns, label="IS-DTS (distributed parallel)", linewidth=2)
    plt.plot(times_s, ptp_errors_ns, label="Simplified PTP (serial master-slave)", linewidth=2)
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Max time difference across healthy nodes (ns)")
    plt.title("Satellite time synchronization convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
