"""Plotting helpers for the IS-DTS satellite synchronization simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Registers the 3D projection.


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_figure(fig: plt.Figure, save_path: str | Path | None, *, dpi: int = 300) -> None:
    """Save a figure if a destination was supplied."""

    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _as_2d_time_satellite(data: np.ndarray | list[list[float]], name: str) -> np.ndarray:
    """Return plot data in ``(time, satellite_or_run)`` order."""

    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array shaped as (time, satellite)")
    return array


def _plot_waterfall_lines(
    ax: Any,
    time: np.ndarray,
    errors: np.ndarray,
    *,
    stride: int = 1,
    line_cmap: str = "coolwarm",
    y_label: str = "Satellite / run index",
    z_label: str = "Time Difference [s]",
) -> None:
    """Draw a clean 3D stacked-line/waterfall convergence view."""

    satellite_count = errors.shape[1]
    selected = np.arange(0, satellite_count, max(1, stride), dtype=int)
    colors = plt.get_cmap(line_cmap)(np.linspace(0.1, 0.9, len(selected)))

    for color, sat_index in zip(colors, selected):
        ax.plot(
            time,
            np.full_like(time, sat_index, dtype=float),
            errors[:, sat_index],
            color=color,
            linewidth=1.2,
            alpha=0.92,
        )

    mean_error = np.nanmean(errors, axis=1)
    ax.plot(
        time,
        np.full_like(time, satellite_count, dtype=float),
        mean_error,
        color="black",
        linewidth=1.8,
        linestyle="--",
        label="Average",
    )
    ax.set_xlabel("Time [s]", labelpad=8)
    ax.set_ylabel(y_label, labelpad=8)
    ax.set_zlabel(z_label, labelpad=8)
    ax.view_init(elev=25, azim=-58)
    ax.grid(True, alpha=0.25)


def _orbit_styles(orbit_count: int) -> tuple[np.ndarray, list[str]]:
    colors = plt.get_cmap("tab10")(np.arange(orbit_count) % 10)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    return colors, markers


def plot_isdts_convergence(
    time: np.ndarray | list[float],
    time_errors: np.ndarray | list[list[float]],
    save_path: str | Path | None = None,
    *,
    title: str = "IS-DTS convergence results",
    satellite_stride: int = 1,
) -> plt.Figure:
    """Create a 3D waterfall plot of IS-DTS time-error convergence.

    Parameters
    ----------
    time:
        One-dimensional time axis in seconds.
    time_errors:
        Two-dimensional time synchronization error array in seconds with shape
        ``(time, satellite_or_run)``.
    save_path:
        Optional destination. When supplied, the figure is written at 300 dpi.
    title:
        Figure title.
    satellite_stride:
        Plot every Nth satellite/run to keep dense constellations readable.
    """

    time_array = np.asarray(time, dtype=float)
    errors = _as_2d_time_satellite(time_errors, "time_errors")

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    _plot_waterfall_lines(ax, time_array, errors, stride=satellite_stride)
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save_figure(fig, save_path)
    return fig


def plot_satellite_accuracy_polar(
    final_errors: np.ndarray | list[float],
    orbit_indices: np.ndarray | list[int] | None = None,
    save_path: str | Path | None = None,
    *,
    title: str = "Satellite time accuracy after convergence",
    failed_satellites: list[int] | tuple[int, ...] | None = None,
) -> plt.Figure:
    """Create a polar time-accuracy plot grouped by orbital plane.

    Radial values are absolute final time differences/errors in seconds. Each
    satellite is placed at an angular location according to its orbital plane
    and in-plane index.
    """

    errors = np.asarray(final_errors, dtype=float)
    satellite_count = errors.size
    if orbit_indices is None:
        orbit_count = 6 if satellite_count % 6 == 0 else max(1, min(6, satellite_count))
        orbit_indices_array = np.arange(satellite_count) % orbit_count
    else:
        orbit_indices_array = np.asarray(orbit_indices, dtype=int)
        if orbit_indices_array.size != satellite_count:
            raise ValueError("orbit_indices must have the same length as final_errors")
        orbit_count = int(np.nanmax(orbit_indices_array)) + 1 if satellite_count else 0

    failed = set(failed_satellites or [])
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    colors, markers = _orbit_styles(orbit_count)
    max_radius = float(np.nanmax(np.abs(errors))) if np.any(np.isfinite(errors)) else 1.0

    for orbit in range(orbit_count):
        indexes = np.where(orbit_indices_array == orbit)[0]
        if indexes.size == 0:
            continue
        theta = 2.0 * np.pi * (orbit + (np.arange(indexes.size) + 0.5) / indexes.size) / orbit_count
        radius = np.abs(errors[indexes])
        ax.scatter(
            theta,
            radius,
            color=colors[orbit],
            marker=markers[orbit % len(markers)],
            s=38,
            alpha=0.9,
            label=f"Orbit plane {orbit + 1}",
        )
        for angle, radial_value, sat_index in zip(theta, radius, indexes):
            if sat_index in failed:
                failed_radius = radial_value if np.isfinite(radial_value) else max_radius * 0.08
                ax.scatter(angle, max(failed_radius, max_radius * 0.08), marker="x", s=95, color="red", linewidth=2.0)

    if failed:
        ax.scatter([], [], marker="x", s=95, color="red", linewidth=2.0, label="Failed satellite")
    ax.set_title(title, pad=18)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8)
    _save_figure(fig, save_path)
    return fig


def plot_isdts_results(
    time: np.ndarray | list[float],
    time_errors: np.ndarray | list[list[float]],
    orbit_indices: np.ndarray | list[int] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create Figure 6-style IS-DTS convergence and final polar accuracy plots."""

    time_array = np.asarray(time, dtype=float)
    errors = _as_2d_time_satellite(time_errors, "time_errors")
    fig = plt.figure(figsize=(13, 6))
    ax_3d = fig.add_subplot(121, projection="3d")
    _plot_waterfall_lines(ax_3d, time_array, errors, stride=max(1, errors.shape[1] // 24))
    ax_3d.set_title("(a) IS-DTS convergence")
    ax_3d.legend(loc="upper right", fontsize=8)

    ax_polar = fig.add_subplot(122, projection="polar")
    _draw_polar_on_axis(ax_polar, errors[-1], orbit_indices, "(b) Time accuracy after convergence")
    fig.suptitle("Figure 6 — IS-DTS simulation results", y=0.98)
    _save_figure(fig, save_path)
    return fig


def _draw_polar_on_axis(
    ax: Any,
    final_errors: np.ndarray,
    orbit_indices: np.ndarray | list[int] | None,
    title: str,
    failed_satellites: list[int] | tuple[int, ...] | None = None,
) -> None:
    """Draw the polar accuracy graphic on a caller-provided axis."""

    errors = np.asarray(final_errors, dtype=float)
    satellite_count = errors.size
    if orbit_indices is None:
        orbit_count = 6 if satellite_count % 6 == 0 else max(1, min(6, satellite_count))
        orbit_indices_array = np.arange(satellite_count) % orbit_count
    else:
        orbit_indices_array = np.asarray(orbit_indices, dtype=int)
        orbit_count = int(np.nanmax(orbit_indices_array)) + 1 if satellite_count else 0
    colors, markers = _orbit_styles(orbit_count)
    failed = set(failed_satellites or [])
    max_radius = float(np.nanmax(np.abs(errors))) if np.any(np.isfinite(errors)) else 1.0

    for orbit in range(orbit_count):
        indexes = np.where(orbit_indices_array == orbit)[0]
        if indexes.size == 0:
            continue
        theta = 2.0 * np.pi * (orbit + (np.arange(indexes.size) + 0.5) / indexes.size) / orbit_count
        radius = np.abs(errors[indexes])
        ax.scatter(
            theta,
            radius,
            color=colors[orbit],
            marker=markers[orbit % len(markers)],
            s=32,
            alpha=0.9,
            label=f"Orbit {orbit + 1}",
        )
        for angle, radial_value, sat_index in zip(theta, radius, indexes):
            if sat_index in failed:
                failed_radius = radial_value if np.isfinite(radial_value) else max_radius * 0.08
                ax.scatter(angle, max(failed_radius, max_radius * 0.08), marker="x", s=90, color="red", linewidth=2.0)

    if failed:
        ax.scatter([], [], marker="x", s=90, color="red", linewidth=2.0, label="Failed")
    ax.set_title(title, pad=16)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=7)


def plot_peak_to_peak_difference(
    time: np.ndarray | list[float],
    diff_by_interval: dict[float, np.ndarray | list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot peak-to-peak time difference versus time for adjustment intervals."""

    time_array = np.asarray(time, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for interval, values in sorted(diff_by_interval.items()):
        diff = np.asarray(values, dtype=float)
        ax.plot(time_array, np.maximum(diff, np.finfo(float).tiny), linewidth=1.8, label=f"interval = {interval:g} s")
    ax.set_yscale("log")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Peak-to-peak Time Difference [s]")
    ax.set_title("Figure 7 — Peak-to-peak time difference over time")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    _save_figure(fig, save_path)
    return fig


def plot_traditional_ptp_performance(
    time: np.ndarray | list[float],
    same_orbit_data: np.ndarray | list[list[float]],
    different_orbit_data: np.ndarray | list[list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create Figure 8-style traditional PTP performance subplots."""

    time_array = np.asarray(time, dtype=float)
    same_orbit = _as_2d_time_satellite(same_orbit_data, "same_orbit_data")
    different_orbit = _as_2d_time_satellite(different_orbit_data, "different_orbit_data")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for sat_index in range(same_orbit.shape[1]):
        axes[0].plot(time_array, same_orbit[:, sat_index], linewidth=0.8, alpha=0.75)
    axes[0].set_title("(a) Same orbital plane")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Time Difference [s]")
    axes[0].grid(True, alpha=0.3)

    for orbit_index in range(different_orbit.shape[1]):
        axes[1].plot(time_array, different_orbit[:, orbit_index], linewidth=1.5, label=f"Orbit plane {orbit_index + 1}")
    axes[1].set_title("(b) Different orbital planes")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Time Difference [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("Figure 8 — Traditional PTP performance")
    _save_figure(fig, save_path)
    return fig


def plot_robustness_polar(
    final_errors: np.ndarray | list[float],
    orbit_indices: np.ndarray | list[int] | None = None,
    failed_satellites: list[int] | tuple[int, ...] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a robustness polar plot with optional failed satellite markers."""

    return plot_satellite_accuracy_polar(
        final_errors,
        orbit_indices,
        save_path,
        title="Robustness accuracy with satellite-node failures",
        failed_satellites=failed_satellites,
    )


def plot_robustness_convergence_3d(
    time: np.ndarray | list[float],
    time_errors: np.ndarray | list[list[float]],
    save_path: str | Path | None = None,
    *,
    failed_satellites: list[int] | tuple[int, ...] | None = None,
) -> plt.Figure:
    """Create a 3D convergence plot for a satellite-node-failure scenario."""

    fig = plot_isdts_convergence(
        time,
        time_errors,
        save_path=None,
        title="Robustness convergence with satellite-node failures",
        satellite_stride=max(1, np.asarray(time_errors).shape[1] // 24),
    )
    if failed_satellites:
        fig.axes[0].text2D(0.02, 0.95, f"Failed satellites: {list(failed_satellites)}", transform=fig.axes[0].transAxes)
    _save_figure(fig, save_path)
    return fig


def plot_robustness_results(
    time: np.ndarray | list[float],
    time_errors: np.ndarray | list[list[float]],
    orbit_indices: np.ndarray | list[int] | None = None,
    failed_satellites: list[int] | tuple[int, ...] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create Figure 9-style polar and 3D robustness-result subplots."""

    time_array = np.asarray(time, dtype=float)
    errors = _as_2d_time_satellite(time_errors, "time_errors")
    fig = plt.figure(figsize=(13, 6))
    ax_polar = fig.add_subplot(121, projection="polar")
    _draw_polar_on_axis(ax_polar, errors[-1], orbit_indices, "(a) Accuracy with two node failures", failed_satellites)
    ax_3d = fig.add_subplot(122, projection="3d")
    _plot_waterfall_lines(
        ax_3d,
        time_array,
        errors,
        stride=max(1, errors.shape[1] // 24),
        y_label="Satellite / orbit index",
    )
    ax_3d.set_title("(b) Robust IS-DTS convergence")
    if failed_satellites:
        ax_3d.text2D(0.02, 0.95, f"Failed satellites: {list(failed_satellites)}", transform=ax_3d.transAxes)
    fig.suptitle("Figure 9 — Robustness results of IS-DTS", y=0.98)
    _save_figure(fig, save_path)
    return fig


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
