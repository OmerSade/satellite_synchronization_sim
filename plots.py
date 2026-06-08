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
    z_label: str = "Absolute Time Difference [s]",
) -> None:
    """Draw a clean 3D stacked-line/waterfall convergence view."""

    display_errors = np.abs(errors)
    satellite_count = display_errors.shape[1]
    selected = np.arange(0, satellite_count, max(1, stride), dtype=int)
    colors = plt.get_cmap(line_cmap)(np.linspace(0.1, 0.9, len(selected)))

    for color, sat_index in zip(colors, selected):
        ax.plot(
            time,
            np.full_like(time, sat_index, dtype=float),
            display_errors[:, sat_index],
            color=color,
            linewidth=1.2,
            alpha=0.92,
        )

    mean_error = np.nanmean(display_errors, axis=1)
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


def _resolve_orbit_indices(
    satellite_count: int,
    orbit_indices: np.ndarray | list[int] | None,
    *,
    default_orbit_count: int = 6,
) -> tuple[np.ndarray, int]:
    """Return one orbit-plane index per satellite and the number of planes."""

    if orbit_indices is None:
        if satellite_count % default_orbit_count == 0:
            orbit_count = default_orbit_count
        else:
            orbit_count = max(1, min(default_orbit_count, satellite_count))
        indexes = np.repeat(np.arange(orbit_count), int(np.ceil(satellite_count / orbit_count)))[
            :satellite_count
        ]
    else:
        indexes = np.asarray(orbit_indices, dtype=int)
        if indexes.size != satellite_count:
            raise ValueError("orbit_indices must have the same length as the satellite dimension")
        orbit_count = int(np.nanmax(indexes)) + 1 if satellite_count else 0
    return indexes, orbit_count


def _orbit_convergence_from_satellites(
    time_errors: np.ndarray,
    orbit_indices: np.ndarray | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Average per-orbit convergence errors relative to all satellites.

    Figure 6(a) uses one line per orbit.  At each time step, every satellite's
    value is first referenced to the average value of the full constellation.
    The absolute value of that difference is then averaged across satellites in
    the same orbit, so the trace shows difference magnitude rather than signed
    direction.
    """

    errors = _as_2d_time_satellite(time_errors, "time_errors")
    indexes, orbit_count = _resolve_orbit_indices(errors.shape[1], orbit_indices)
    referenced_errors = np.abs(errors - np.nanmean(errors, axis=1, keepdims=True))
    orbit_errors = np.empty((errors.shape[0], orbit_count), dtype=float)
    for orbit in range(orbit_count):
        orbit_mask = indexes == orbit
        orbit_errors[:, orbit] = np.nanmean(referenced_errors[:, orbit_mask], axis=1)
    return orbit_errors, indexes, orbit_count


def _plot_orbit_waterfall_lines(
    ax: Any,
    time: np.ndarray,
    time_errors: np.ndarray,
    orbit_indices: np.ndarray | list[int] | None = None,
) -> None:
    """Draw Figure 6(a)-style 3D lines where each line is one orbit plane."""

    orbit_errors, _, orbit_count = _orbit_convergence_from_satellites(time_errors, orbit_indices)
    colors, markers = _orbit_styles(orbit_count)
    for orbit in range(orbit_count):
        y_value = orbit + 1
        ax.plot(
            time,
            np.full_like(time, y_value, dtype=float),
            orbit_errors[:, orbit],
            color=colors[orbit],
            linewidth=1.8,
            marker=markers[orbit % len(markers)],
            markevery=max(1, len(time) // 8),
            markersize=3.5,
            label=f"Orbit {y_value}",
        )
    ax.plot(
        time,
        np.full_like(time, orbit_count + 1, dtype=float),
        np.zeros_like(time, dtype=float),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="72-satellite average",
    )
    ax.set_xlabel("Time [s]", labelpad=8)
    ax.set_ylabel("Orbit plane", labelpad=8)
    ax.set_zlabel("Absolute time error vs. 72-satellite average [s]", labelpad=8)
    ax.set_yticks(np.arange(1, orbit_count + 1))
    ax.set_yticklabels([f"Orbit {orbit + 1}" for orbit in range(orbit_count)])
    ax.view_init(elev=25, azim=-58)
    ax.grid(True, alpha=0.25)


def plot_isdts_convergence(
    time: np.ndarray | list[float],
    time_errors: np.ndarray | list[list[float]],
    save_path: str | Path | None = None,
    *,
    title: str = "IS-DTS convergence results",
    satellite_stride: int = 1,
    orbit_indices: np.ndarray | list[int] | None = None,
) -> plt.Figure:
    """Create a 3D orbit-level waterfall plot of IS-DTS convergence.

    Parameters
    ----------
    time:
        One-dimensional time axis in seconds.
    time_errors:
        Two-dimensional time synchronization error array in seconds with shape
        ``(time, satellite)``.  Figure 6(a) is computed from this data as one
        trace per orbit plane, referenced to the average of all satellites at
        each time step.
    save_path:
        Optional destination. When supplied, the figure is written at 300 dpi.
    title:
        Figure title.
    satellite_stride:
        Deprecated compatibility argument. Orbit-level Figure 6 plotting no
        longer strides satellite traces because each line is an orbit.
    orbit_indices:
        Optional orbit-plane index for each satellite.  If omitted, satellites
        are split into six equally sized orbit planes when possible.
    """

    del satellite_stride  # Kept for backward-compatible callers.
    time_array = np.asarray(time, dtype=float)
    errors = _as_2d_time_satellite(time_errors, "time_errors")

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    _plot_orbit_waterfall_lines(ax, time_array, errors, orbit_indices)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
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

    Radial values are absolute final time differences/errors in seconds. All
    satellites in the same orbital plane share that orbit's polar angle.
    """

    errors = np.asarray(final_errors, dtype=float)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    _draw_polar_on_axis(ax, errors, orbit_indices, title, failed_satellites)
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
    _plot_orbit_waterfall_lines(ax_3d, time_array, errors, orbit_indices)
    ax_3d.set_title("(a) Orbit convergence vs. 72-satellite average")
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
    """Draw final satellite accuracy with one polar angle per orbit plane.

    Figure 6(b) places all satellites from the same orbit on the same angular
    spoke.  The radial coordinate is each satellite's absolute time accuracy
    after convergence, so the 72 satellite accuracies are still visible while
    the six angular positions identify the six orbit planes.
    """

    errors = np.asarray(final_errors, dtype=float)
    satellite_count = errors.size
    orbit_indices_array, orbit_count = _resolve_orbit_indices(satellite_count, orbit_indices)
    colors, markers = _orbit_styles(orbit_count)
    failed = set(failed_satellites or [])
    max_radius = float(np.nanmax(np.abs(errors))) if np.any(np.isfinite(errors)) else 1.0
    orbit_angles = 2.0 * np.pi * np.arange(orbit_count) / max(orbit_count, 1)

    for orbit in range(orbit_count):
        indexes = np.where(orbit_indices_array == orbit)[0]
        if indexes.size == 0:
            continue
        theta = np.full(indexes.size, orbit_angles[orbit], dtype=float)
        radius = np.abs(errors[indexes])
        ax.scatter(
            theta,
            radius,
            color=colors[orbit],
            marker=markers[orbit % len(markers)],
            s=36,
            alpha=0.88,
            label=f"Orbit {orbit + 1}",
        )
        finite_radius = radius[np.isfinite(radius)]
        if finite_radius.size:
            ax.scatter(
                orbit_angles[orbit],
                float(np.mean(finite_radius)),
                color=colors[orbit],
                edgecolor="black",
                marker="D",
                s=72,
                linewidth=1.0,
            )
        for radial_value, sat_index in zip(radius, indexes):
            if sat_index in failed:
                failed_radius = radial_value if np.isfinite(radial_value) else max_radius * 0.08
                ax.scatter(
                    orbit_angles[orbit],
                    max(failed_radius, max_radius * 0.08),
                    marker="x",
                    s=90,
                    color="red",
                    linewidth=2.0,
                )

    if failed:
        ax.scatter([], [], marker="x", s=90, color="red", linewidth=2.0, label="Failed")
    ax.scatter([], [], color="white", edgecolor="black", marker="D", s=72, linewidth=1.0, label="Orbit mean")
    ax.set_title(title, pad=16)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_xticks(orbit_angles)
    ax.set_xticklabels([f"Orbit {orbit + 1}" for orbit in range(orbit_count)])
    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=7)


def plot_peak_to_peak_difference(
    time: np.ndarray | list[float],
    diff_by_interval: dict[float, np.ndarray | list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot peak-to-peak time difference versus time for adjustment intervals."""

    time_array = np.asarray(time, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for interval, values in sorted(diff_by_interval.items()):
        diff = np.abs(np.asarray(values, dtype=float))
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
        axes[0].plot(time_array, np.abs(same_orbit[:, sat_index]), linewidth=0.8, alpha=0.75)
    axes[0].set_title("(a) Same orbital plane")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Absolute Time Difference [s]")
    axes[0].grid(True, alpha=0.3)

    for orbit_index in range(different_orbit.shape[1]):
        axes[1].plot(
            time_array,
            np.abs(different_orbit[:, orbit_index]),
            linewidth=1.5,
            label=f"Orbit plane {orbit_index + 1}",
        )
    axes[1].set_title("(b) Different orbital planes")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Absolute Time Difference [s]")
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
