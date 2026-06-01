"""Plotting helpers for the IS-DTS satellite synchronization simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_error_evolution(history: dict[str, Any], output_dir: str | Path) -> Path:
    """Plot peak-to-peak and RMS clock error versus synchronization round."""

    output_dir = Path(output_dir)
    steps = np.asarray(history["step"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, history["peak_to_peak_error_ns"], label="Peak-to-peak error")
    ax.plot(steps, history["rms_error_ns"], label="RMS error")
    ax.set_xlabel("Synchronization round")
    ax.set_ylabel("Clock error [ns]")
    ax.set_title("Distributed inter-satellite synchronization convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, output_dir, "error_evolution.png")


def plot_clock_offsets(history: dict[str, Any], output_dir: str | Path) -> Path:
    """Plot clock offsets for all satellites over time."""

    output_dir = Path(output_dir)
    offsets = np.asarray(history["clock_offsets_ns"])
    fig, ax = plt.subplots(figsize=(9, 5))
    for satellite_index in range(offsets.shape[1]):
        ax.plot(history["step"], offsets[:, satellite_index], linewidth=0.9, alpha=0.75)
    ax.set_xlabel("Synchronization round")
    ax.set_ylabel("Clock offset [ns]")
    ax.set_title("Satellite clock offsets during IS-DTS consensus")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir, "clock_offsets.png")


def plot_link_quality(history: dict[str, Any], output_dir: str | Path) -> Path:
    """Plot RF/laser measurement quality produced by the communication modules."""

    output_dir = Path(output_dir)
    steps = np.asarray(history["step"])
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(steps, history["mean_laser_snr_db"], color="tab:blue")
    axes[0].set_ylabel("Laser SNR [dB]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, history["mean_rf_ebno_db"], color="tab:orange")
    axes[1].set_ylabel("RF Eb/No [dB]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, history["active_measurements"], color="tab:green")
    axes[2].set_xlabel("Synchronization round")
    axes[2].set_ylabel("Accepted measurements")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Inter-satellite link quality and accepted exchanges")
    return _save(fig, output_dir, "link_quality.png")


def plot_monte_carlo_summary(summary: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    """Create aggregate plots across Monte Carlo draws."""

    output_dir = Path(output_dir)
    final_rms = np.asarray([run["final_rms_error_ns"] for run in summary], dtype=float)
    convergence = np.asarray(
        [np.nan if run["convergence_step"] is None else run["convergence_step"] for run in summary],
        dtype=float,
    )

    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(final_rms, bins=min(12, max(3, len(final_rms))), color="tab:purple", alpha=0.8)
    ax.set_xlabel("Final RMS error [ns]")
    ax.set_ylabel("Monte Carlo draws")
    ax.set_title("Monte Carlo final synchronization accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    paths.append(_save(fig, output_dir, "monte_carlo_final_rms_histogram.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(np.arange(len(summary)), final_rms, label="Final RMS error", color="tab:purple")
    if not np.all(np.isnan(convergence)):
        ax2 = ax.twinx()
        ax2.scatter(np.arange(len(summary)), convergence, label="Convergence step", color="tab:cyan")
        ax2.set_ylabel("Convergence step")
    ax.set_xlabel("Monte Carlo draw")
    ax.set_ylabel("Final RMS error [ns]")
    ax.set_title("Monte Carlo draw outcomes")
    ax.grid(True, alpha=0.3)
    paths.append(_save(fig, output_dir, "monte_carlo_outcomes.png"))
    return paths


def create_run_plots(history: dict[str, Any], output_dir: str | Path) -> list[Path]:
    """Create all per-run plots used by the simulation."""

    return [
        plot_error_evolution(history, output_dir),
        plot_clock_offsets(history, output_dir),
        plot_link_quality(history, output_dir),
    ]
