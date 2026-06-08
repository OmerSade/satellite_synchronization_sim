"""Inter-satellite distributed time synchronization simulation.

This script implements a compact educational version of the IS-DTS workflow
inspired by "Inter-satellite distributed time synchronization solution with
nanosecond accuracy in satellite networks".  It keeps all tunable values in the
root-level ``parameters`` JSON file, uses the repository RF and laser receiver
modules for every timestamp exchange, and writes a self-contained output folder
for each regular or Monte Carlo run.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
DEFAULT_PARAMETERS_FILE = Path(__file__).with_name("parameters")


class SatelliteState(Enum):
    """Minimal satellite state machine expected by the communication modules."""

    INITIALIZING = "Initializing"
    LISTENING = "Listening"
    ADJUSTING = "Adjusting"
    FAULTY = "Faulty"


@dataclass
class Satellite:
    """Clock model for one satellite in the distributed synchronization mesh."""

    node_id: int
    offset_ns: float
    drift_ns_per_s: float
    orbital_plane: int = 0
    satellite_index: int = 0
    state: SatelliteState = SatelliteState.INITIALIZING

    def complete_initialization(self) -> None:
        if self.state != SatelliteState.FAULTY:
            self.state = SatelliteState.LISTENING

    def tick(self, dt_s: float) -> None:
        if self.state != SatelliteState.FAULTY:
            self.offset_ns += self.drift_ns_per_s * dt_s

    def local_time_ns(self, true_time_s: float) -> float:
        return true_time_s * 1e9 + self.offset_ns

    def apply_correction(self, correction_ns: float) -> None:
        if self.state != SatelliteState.FAULTY:
            self.state = SatelliteState.ADJUSTING
            self.offset_ns += correction_ns
            self.state = SatelliteState.LISTENING

    @property
    def is_healthy(self) -> bool:
        return self.state != SatelliteState.FAULTY


def _install_satellite_compatibility_module() -> None:
    """Expose the local Satellite class under ``src.satellite`` for legacy files."""

    src_module = sys.modules.setdefault("src", types.ModuleType("src"))
    satellite_module = types.ModuleType("src.satellite")
    satellite_module.Satellite = Satellite
    satellite_module.SatelliteState = SatelliteState
    sys.modules["src.satellite"] = satellite_module
    setattr(src_module, "satellite", satellite_module)


_install_satellite_compatibility_module()

from satellite_laser_communication import (  # noqa: E402
    LaserLinkConfig,
    LaserLinkGeometry,
    SatelliteLaserCommunication,
)
from satellite_receiver import RFLinkGeometry, RFReceiverConfig, SatelliteRFReceiver  # noqa: E402


def load_parameters(path: str | Path = DEFAULT_PARAMETERS_FILE) -> dict[str, Any]:
    """Read the root-level JSON parameter file before a simulation starts."""

    parameter_path = Path(path)
    with parameter_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_parameters(config: dict[str, Any], path: str | Path) -> None:
    """Persist a parameter snapshot for reproducible run folders."""

    with Path(path).open("w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def nested_get(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        value = value[part]
    return value


def nested_set(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


def sample_monte_carlo_value(spec: dict[str, Any], rng: np.random.Generator) -> float:
    """Draw one parameter value from a Monte Carlo distribution spec."""

    distribution = spec["distribution"].lower()
    if distribution == "uniform":
        value = rng.uniform(float(spec["low"]), float(spec["high"]))
    elif distribution == "normal":
        value = rng.normal(float(spec["mean"]), float(spec["std"]))
    elif distribution == "loguniform":
        value = np.exp(rng.uniform(np.log(float(spec["low"])), np.log(float(spec["high"]))))
    else:
        raise ValueError(f"Unsupported Monte Carlo distribution: {distribution}")

    if "min" in spec:
        value = max(value, float(spec["min"]))
    if "max" in spec:
        value = min(value, float(spec["max"]))
    return float(value)


def time_sync_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the required paper-aligned IS-DTS parameter section."""

    return config["is_dts_simulation"]


def simulation_time_step_s(config: dict[str, Any]) -> float:
    return float(time_sync_config(config)["des_time_step_s"])


def adjustment_interval_s(config: dict[str, Any]) -> float:
    sync = time_sync_config(config)
    return float(sync.get("time_adjustment_interval_s", simulation_time_step_s(config)))


def simulation_duration_s(config: dict[str, Any]) -> float:
    sync = time_sync_config(config)
    if "duration_s" in sync:
        return float(sync["duration_s"])
    return int(sync["steps"]) * simulation_time_step_s(config)


def simulation_step_count(config: dict[str, Any]) -> int:
    sync = time_sync_config(config)
    if "steps" in sync:
        return int(sync["steps"])
    return max(1, int(round(simulation_duration_s(config) / simulation_time_step_s(config))))


def paper_satellite_count(config: dict[str, Any]) -> int:
    return int(time_sync_config(config)["satellite_count"])


def create_satellites(config: dict[str, Any], rng: np.random.Generator) -> list[Satellite]:
    sync = time_sync_config(config)
    count = paper_satellite_count(config)
    satellites_per_plane = int(sync["satellites_per_plane"])

    max_initial_offset_s = float(sync["max_initial_time_offset_s"])
    oscillator_accuracy = float(sync["oscillator_frequency_accuracy"])
    satellites: list[Satellite] = []
    for node_id in range(count):
        # Paper parameter: initial satellite clock offsets are uniformly
        # distributed within ±1 s for the IS-DTS DES experiments.
        offset_ns = float(
            rng.uniform(-max_initial_offset_s, max_initial_offset_s) * 1e9
        )

        # Fractional oscillator frequency accuracy maps directly to clock
        # drift in seconds per second; store as ns/s for the local model.
        drift_ns_per_s = float(
            rng.uniform(-oscillator_accuracy, oscillator_accuracy) * 1e9
        )

        satellites.append(
            Satellite(
                node_id=node_id,
                offset_ns=offset_ns,
                drift_ns_per_s=drift_ns_per_s,
                orbital_plane=node_id // satellites_per_plane,
                satellite_index=node_id % satellites_per_plane,
            )
        )

    for node_id in sync.get("faulty_nodes", []):
        satellites[int(node_id)].state = SatelliteState.FAULTY
    return satellites


def _walker_node_id(plane: int, sat_index: int, satellites_per_plane: int) -> int:
    return plane * satellites_per_plane + sat_index


def build_walker_mesh_topology(config: dict[str, Any]) -> list[tuple[int, int, str]]:
    """Build the deterministic paper-style Walker mesh ISL topology.

    Each satellite has two same-plane neighbors and two cross-plane neighbors in
    neighboring planes moving in the same direction.  Inter-plane polar-region
    disconnections are applied dynamically during the simulation.
    """

    sync = time_sync_config(config)
    planes = int(sync["orbital_planes"])
    satellites_per_plane = int(sync["satellites_per_plane"])
    edges: set[tuple[int, int, str]] = set()

    for plane in range(planes):
        for sat_index in range(satellites_per_plane):
            node_id = _walker_node_id(plane, sat_index, satellites_per_plane)
            ahead = _walker_node_id(
                plane, (sat_index + 1) % satellites_per_plane, satellites_per_plane
            )
            edges.add((min(node_id, ahead), max(node_id, ahead), "same_plane"))

            if int(sync.get("cross_plane_neighbors", 2)) > 0:
                for neighbor_plane in ((plane - 1) % planes, (plane + 1) % planes):
                    # The paper excludes opposite-direction orbital-plane ISLs.
                    # This Walker-like educational model treats adjacent planes
                    # as same-direction and keeps the legacy random topology out
                    # of the paper-based IS-DTS path.
                    other = _walker_node_id(neighbor_plane, sat_index, satellites_per_plane)
                    edges.add((min(node_id, other), max(node_id, other), "cross_plane"))

    return sorted(edges)


def build_topology(config: dict[str, Any]) -> list[tuple[int, int, str]]:
    """Build the deterministic paper Walker mesh topology."""

    return build_walker_mesh_topology(config)


def satellite_latitude_deg(satellite: Satellite, true_time_s: float, config: dict[str, Any]) -> float:
    """Approximate Walker-orbit latitude for polar ISL gating.

    This compact DES model uses one sinusoidal orbit phase per satellite; it is
    sufficient to apply the paper's latitude > 66.5° inter-plane disconnection
    rule without introducing a full orbit propagator.
    """

    sync = time_sync_config(config)
    satellites_per_plane = int(sync.get("satellites_per_plane", max(1, paper_satellite_count(config))))
    inclination_deg = float(sync.get("inclination_deg", 90.0))
    orbital_period_s = float(sync.get("orbital_period_s", 6307.0))
    phase = 2.0 * np.pi * (
        satellite.satellite_index / satellites_per_plane + true_time_s / orbital_period_s
    )
    return float(inclination_deg * np.sin(phase))


def is_edge_active(
    left: int,
    right: int,
    edge_type: str,
    satellites: list[Satellite],
    true_time_s: float,
    config: dict[str, Any],
) -> bool:
    """Return whether an ISL is active after health and polar-region checks."""

    sat_left = satellites[left]
    sat_right = satellites[right]
    if not sat_left.is_healthy or not sat_right.is_healthy:
        return False

    sync = time_sync_config(config)
    if edge_type == "cross_plane" and bool(sync.get("polar_cross_plane_disconnect", False)):
        threshold = float(sync.get("polar_disconnect_latitude_deg", 90.0))
        if (
            abs(satellite_latitude_deg(sat_left, true_time_s, config)) > threshold
            or abs(satellite_latitude_deg(sat_right, true_time_s, config)) > threshold
        ):
            return False
    return True


def paper_failure_node_id(event: dict[str, Any], config: dict[str, Any]) -> int:
    """Translate paper scenario plane/satellite numbering to zero-based node id."""

    if "node_id" in event:
        return int(event["node_id"])
    sync = time_sync_config(config)
    satellites_per_plane = int(sync.get("satellites_per_plane", 1))
    return _walker_node_id(
        int(event["orbital_plane"]) - 1,
        int(event["satellite_index"]) - 1,
        satellites_per_plane,
    )


def update_satellite_failures(
    satellites: list[Satellite], config: dict[str, Any], true_time_s: float, step: int
) -> None:
    """Apply paper time-window node failures from ``is_dts_simulation`` only."""

    sync = time_sync_config(config)
    active_failures: set[int] = {int(node_id) for node_id in sync.get("faulty_nodes", [])}

    for event in sync.get("failure_scenarios", []):
        start_s = float(event["failure_start_s"])
        duration_s = float(event["failure_duration_s"])
        if start_s <= true_time_s < start_s + duration_s:
            active_failures.add(paper_failure_node_id(event, config))

    for satellite in satellites:
        if satellite.node_id in active_failures:
            satellite.state = SatelliteState.FAULTY
        elif satellite.state == SatelliteState.FAULTY:
            satellite.state = SatelliteState.LISTENING


def make_laser_config(config: dict[str, Any]) -> LaserLinkConfig:
    return LaserLinkConfig(**config["laser"])


def make_rf_config(config: dict[str, Any]) -> RFReceiverConfig:
    return RFReceiverConfig(**config["rf"])


def link_geometry(config: dict[str, Any], rng: np.random.Generator) -> tuple[LaserLinkGeometry, RFLinkGeometry]:
    geometry = time_sync_config(config)["link_geometry"]
    range_m = max(
        1.0,
        float(geometry["base_range_m"])
        + float(rng.normal(0.0, geometry["range_jitter_m"])),
    )
    laser_geometry = LaserLinkGeometry(
        range_m=range_m,
        elevation_deg=float(geometry["elevation_deg"]),
        pointing_error_rad=float(geometry["pointing_error_rad"]),
        zenith_transmittance=float(geometry["zenith_transmittance"]),
    )
    rf_geometry = RFLinkGeometry(
        range_m=range_m,
        relative_velocity_m_per_s=float(
            rng.normal(0.0, geometry["relative_velocity_std_m_per_s"])
        ),
    )
    return laser_geometry, rf_geometry


def run_single_simulation(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """Run one distributed synchronization experiment and write outputs."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_parameters(config, run_dir / "parameters.json")

    sync = time_sync_config(config)
    dt_s = simulation_time_step_s(config)
    adjust_interval_s = adjustment_interval_s(config)
    adjust_every_steps = max(1, int(round(adjust_interval_s / dt_s)))
    total_steps = simulation_step_count(config)
    rng = np.random.default_rng(int(sync["seed"]))
    satellites = create_satellites(config, rng)
    edges = build_topology(config)
    laser_config = make_laser_config(config)
    rf_config = make_rf_config(config)

    history: dict[str, Any] = {
        "step": [],
        "clock_offsets_ns": [],
        "peak_to_peak_error_ns": [],
        "rms_error_ns": [],
        "mean_laser_snr_db": [],
        "mean_rf_ebno_db": [],
        "active_measurements": [],
        "per_orbital_plane_time_difference_ns": [],
    }

    for step in range(total_steps):
        true_time_s = step * dt_s
        update_satellite_failures(satellites, config, true_time_s, step)
        for satellite in satellites:
            satellite.tick(dt_s)

        corrections: dict[int, list[float]] = {sat.node_id: [] for sat in satellites if sat.is_healthy}
        laser_snr_values: list[float] = []
        rf_ebno_values: list[float] = []
        accepted_measurements = 0

        if step % adjust_every_steps == 0:
            for left, right, edge_type in edges:
                if rng.random() < float(sync.get("link_dropout_probability", 0.0)):
                    continue
                if not is_edge_active(left, right, edge_type, satellites, true_time_s, config):
                    continue
                sat_left = satellites[left]
                sat_right = satellites[right]

                laser_geometry, rf_geometry = link_geometry(config, rng)

                send_timing_noise_ns = (
                    float(sync.get("timing_difference_sending_messages_s", 0.0)) * 1e9
                )

                laser_receiver = SatelliteLaserCommunication(
                    receiver=sat_right, config=laser_config, rng=rng
                )
                laser_measurement = laser_receiver.measure_downlink(sat_left, true_time_s, laser_geometry)
                laser_snr_values.append(laser_measurement.snr_db)
                if laser_measurement.acquired:
                    estimated_offset_ns = laser_measurement.estimated_offset_ns
                    if send_timing_noise_ns:
                        estimated_offset_ns += float(rng.uniform(-send_timing_noise_ns, send_timing_noise_ns))
                    corrections[right].append(-float(sync["laser_weight"]) * estimated_offset_ns)
                    accepted_measurements += 1

                rf_receiver = SatelliteRFReceiver(receiver=sat_left, config=rf_config, rng=rng)
                rf_measurement = rf_receiver.receive_time_frame(sat_right, true_time_s, rf_geometry)
                rf_ebno_values.append(rf_measurement.ebno_db)
                if rf_measurement.synchronized:
                    estimated_offset_ns = rf_measurement.estimated_offset_ns
                    if send_timing_noise_ns:
                        estimated_offset_ns += float(rng.uniform(-send_timing_noise_ns, send_timing_noise_ns))
                    corrections[left].append(-float(sync["rf_weight"]) * estimated_offset_ns)
                    accepted_measurements += 1

        for node_id, node_corrections in corrections.items():
            if node_corrections:
                satellites[node_id].apply_correction(
                    float(sync["distributed_gain"]) * float(np.mean(node_corrections))
                )

        offsets = np.asarray([sat.offset_ns for sat in satellites], dtype=float)
        healthy_offsets = np.asarray([sat.offset_ns for sat in satellites if sat.is_healthy], dtype=float)
        centered = healthy_offsets - float(np.mean(healthy_offsets)) if healthy_offsets.size else np.array([0.0])

        history["step"].append(step)
        history["clock_offsets_ns"].append(offsets.tolist())
        history["peak_to_peak_error_ns"].append(float(np.ptp(healthy_offsets)) if healthy_offsets.size else 0.0)
        history["rms_error_ns"].append(float(np.sqrt(np.mean(centered**2))))
        history["mean_laser_snr_db"].append(float(np.mean(laser_snr_values)) if laser_snr_values else float("nan"))
        history["mean_rf_ebno_db"].append(float(np.mean(rf_ebno_values)) if rf_ebno_values else float("nan"))
        plane_differences: list[float] = []
        planes = int(sync.get("orbital_planes", 1))
        for plane in range(planes):
            plane_offsets = np.asarray(
                [sat.offset_ns for sat in satellites if sat.is_healthy and sat.orbital_plane == plane],
                dtype=float,
            )
            plane_differences.append(float(np.ptp(plane_offsets)) if plane_offsets.size else float("nan"))

        history["active_measurements"].append(accepted_measurements)
        history["per_orbital_plane_time_difference_ns"].append(plane_differences)

    if sync.get("save_csv", True):
        save_csv_outputs(history, run_dir)
    if sync.get("save_plots", True):
        plots = importlib.import_module("plots")
        plots.create_run_plots(history, run_dir)
        create_simulation_result_plots(history, config, run_dir)

    convergence_step = next(
        (
            int(step)
            for step, rms in zip(history["step"], history["rms_error_ns"])
            if rms <= float(sync["convergence_threshold_ns"])
        ),
        None,
    )
    summary = {
        "run_dir": str(run_dir),
        "final_rms_error_ns": float(history["rms_error_ns"][-1]),
        "final_peak_to_peak_error_ns": float(history["peak_to_peak_error_ns"][-1]),
        "convergence_step": convergence_step,
        "convergence_time_s": None if convergence_step is None else float(convergence_step * dt_s),
        "maximum_time_difference_over_time_ns": float(np.nanmax(history["peak_to_peak_error_ns"])),
        "peak_to_peak_time_difference_after_convergence_ns": (
            float(history["peak_to_peak_error_ns"][convergence_step])
            if convergence_step is not None
            else float(history["peak_to_peak_error_ns"][-1])
        ),
        "per_orbital_plane_time_differences_final_ns": history["per_orbital_plane_time_difference_ns"][-1],
        "accepted_measurements_total": int(np.sum(history["active_measurements"])),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return {"history": history, "summary": summary}





def _orbit_indices(satellite_count: int, orbit_count: int = 6) -> np.ndarray:
    """Assign simulated satellites to equally sized orbital planes for plotting."""

    bounded_orbit_count = max(1, min(int(orbit_count), int(satellite_count)))
    return np.repeat(
        np.arange(bounded_orbit_count),
        int(np.ceil(satellite_count / bounded_orbit_count)),
    )[:satellite_count]


def build_simulation_plot_data(history: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Build plotting arrays only from values produced by the simulation.

    This intentionally avoids bootstrapping or synthesizing fixed reference data.
    Optional PTP arrays are passed through only if a caller or future simulator
    version stores real PTP outputs in ``history``.
    """

    sim = config["simulation"]
    steps = np.asarray(history["step"], dtype=float)
    time = steps * float(sim["time_step_s"])
    offsets_s = np.asarray(history["clock_offsets_ns"], dtype=float) * 1e-9
    satellite_count = offsets_s.shape[1]
    orbit_count = int(sim.get("orbit_plane_count", 6))
    orbit_indices = _orbit_indices(satellite_count, orbit_count)
    peak_to_peak_s = np.asarray(history["peak_to_peak_error_ns"], dtype=float) * 1e-9

    failed_satellites = sorted(
        {
            int(node_id)
            for node_id in sim.get("faulty_nodes", [])
        }
        | {
            int(event["node_id"])
            for event in sim.get("failure_events", [])
            if "node_id" in event
        }
    )

    data: dict[str, Any] = {
        "time": time,
        "time_errors_s": offsets_s,
        "orbit_indices": orbit_indices,
        "peak_to_peak_by_interval": {float(sim["time_step_s"]): peak_to_peak_s},
        "failed_satellites": failed_satellites,
    }

    if "traditional_ptp_same_orbit_s" in history and "traditional_ptp_different_orbit_s" in history:
        data["same_orbit_ptp_s"] = np.asarray(history["traditional_ptp_same_orbit_s"], dtype=float)
        data["different_orbit_ptp_s"] = np.asarray(history["traditional_ptp_different_orbit_s"], dtype=float)

    return data


def create_simulation_result_plots(
    history: dict[str, Any],
    config: dict[str, Any],
    run_dir: str | Path,
) -> list[Path]:
    """Generate publication-style plots that are backed by simulation outputs."""

    plots = importlib.import_module("plots")
    plot_dir = Path(run_dir) / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data = build_simulation_plot_data(history, config)

    figure_specs = [
        (
            "fig6_isdts_results.png",
            plots.plot_isdts_results(
                data["time"],
                data["time_errors_s"],
                data["orbit_indices"],
                save_path=plot_dir / "fig6_isdts_results.png",
            ),
        ),
        (
            "fig7_peak_to_peak_difference.png",
            plots.plot_peak_to_peak_difference(
                data["time"],
                data["peak_to_peak_by_interval"],
                save_path=plot_dir / "fig7_peak_to_peak_difference.png",
            ),
        ),
    ]

    if "same_orbit_ptp_s" in data and "different_orbit_ptp_s" in data:
        figure_specs.append(
            (
                "fig8_traditional_ptp.png",
                plots.plot_traditional_ptp_performance(
                    data["time"],
                    data["same_orbit_ptp_s"],
                    data["different_orbit_ptp_s"],
                    save_path=plot_dir / "fig8_traditional_ptp.png",
                ),
            )
        )

    if data["failed_satellites"]:
        figure_specs.append(
            (
                "fig9_robustness.png",
                plots.plot_robustness_results(
                    data["time"],
                    data["time_errors_s"],
                    data["orbit_indices"],
                    failed_satellites=data["failed_satellites"],
                    save_path=plot_dir / "fig9_robustness.png",
                ),
            )
        )

    # Plotting functions return figures for reuse/display; close them here so
    # batch simulation runs do not accumulate GUI resources.
    import matplotlib.pyplot as plt

    saved_paths: list[Path] = []
    for filename, figure in figure_specs:
        saved_paths.append(plot_dir / filename)
        plt.close(figure)
    return saved_paths


def save_csv_outputs(history: dict[str, Any], run_dir: Path) -> None:
    """Write metrics and per-satellite offsets for downstream analysis."""

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "step",
                "peak_to_peak_error_ns",
                "rms_error_ns",
                "mean_laser_snr_db",
                "mean_rf_ebno_db",
                "active_measurements",
            ]
        )
        for index, step in enumerate(history["step"]):
            writer.writerow(
                [
                    step,
                    history["peak_to_peak_error_ns"][index],
                    history["rms_error_ns"][index],
                    history["mean_laser_snr_db"][index],
                    history["mean_rf_ebno_db"][index],
                    history["active_measurements"][index],
                ]
            )

    offsets = np.asarray(history["clock_offsets_ns"], dtype=float)
    with (run_dir / "clock_offsets.csv").open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["step", *[f"satellite_{index}_offset_ns" for index in range(offsets.shape[1])]])
        for row_index, step in enumerate(history["step"]):
            writer.writerow([step, *offsets[row_index].tolist()])


def create_output_directory(config: dict[str, Any], suffix: str | None = None) -> Path:
    sync = time_sync_config(config)
    root = Path(sync.get("output_root", "outputs"))
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = sync.get("name", "is_dts_simulation")
    run_name = f"{timestamp}_{name}" if suffix is None else f"{timestamp}_{name}_{suffix}"
    run_dir = root / run_name
    counter = 1
    while run_dir.exists():
        run_dir = root / f"{run_name}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True)
    return run_dir


def run_monte_carlo(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run repeated simulations with random draws from the parameter file."""

    mc = time_sync_config(config).get("monte_carlo", {})
    root_run_dir = create_output_directory(config, suffix="monte_carlo")
    write_parameters(config, root_run_dir / "parameters.json")
    rng = np.random.default_rng(int(mc.get("seed", time_sync_config(config)["seed"])))
    summaries: list[dict[str, Any]] = []

    for draw_index in range(int(mc["runs"])):
        draw_config = copy.deepcopy(config)
        draw_config["is_dts_simulation"].setdefault("monte_carlo", {})["enabled"] = False
        draw_values: dict[str, float] = {}
        for key, spec in mc.get("vary", {}).items():
            value = sample_monte_carlo_value(spec, rng)
            nested_set(draw_config, key, value)
            draw_values[key] = value
        draw_config["is_dts_simulation"]["seed"] = int(rng.integers(0, 2**31 - 1))
        draw_dir = root_run_dir / f"draw_{draw_index:03d}"
        draw_dir.mkdir(parents=True, exist_ok=True)
        write_parameters(draw_config, draw_dir / "parameters_draw.json")
        with (draw_dir / "monte_carlo_draw_values.json").open("w", encoding="utf-8") as file_obj:
            json.dump(draw_values, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")
        result = run_single_simulation(draw_config, draw_dir)
        summary = {"draw": draw_index, "draw_values": draw_values, **result["summary"]}
        summaries.append(summary)

    with (root_run_dir / "monte_carlo_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summaries, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    if time_sync_config(config).get("save_plots", True):
        plots = importlib.import_module("plots")
        plots.plot_monte_carlo_summary(summaries, root_run_dir)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IS-DTS satellite synchronization simulation.")
    parser.add_argument("--parameters", default=str(DEFAULT_PARAMETERS_FILE), help="Path to JSON parameters file")
    parser.add_argument("--monte-carlo", action="store_true", help="Force Monte Carlo mode on")
    parser.add_argument("--single-run", action="store_true", help="Force a single run even if parameters enable Monte Carlo")
    parser.add_argument(
        "--scenario",
        choices=[
            "baseline_is_dts",
            "adjustment_interval_comparison",
            "ptp_comparison",
            "polar_topology_robustness",
            "node_failure_robustness",
        ],
        help="Run one paper comparison scenario",
    )
    args = parser.parse_args()

    config = load_parameters(args.parameters)
    if args.monte_carlo:
        time_sync_config(config).setdefault("monte_carlo", {})["enabled"] = True
    if args.single_run:
        time_sync_config(config).setdefault("monte_carlo", {})["enabled"] = False

    if args.scenario == "baseline_is_dts":
        result = baseline_is_dts(config)
        print_summary(result["summary"])
    elif args.scenario == "adjustment_interval_comparison":
        summaries = adjustment_interval_comparison(config)
        for summary in summaries:
            print(
                f"Adjustment interval {summary['adjustment_interval_s']:.1f} s: "
                f"convergence time {summary['convergence_time_s']} s"
            )
    elif args.scenario == "ptp_comparison":
        summary = ptp_comparison(config)
        print(
            "Traditional PTP same-plane peak-to-peak time difference: "
            f"{summary['same_plane_peak_to_peak_time_difference_ns']:.2f} ns"
        )
        print(
            "Traditional PTP different-plane time difference: "
            f"{summary['different_plane_time_difference_s']:.3e} s"
        )
    elif args.scenario == "polar_topology_robustness":
        summaries = polar_topology_robustness(config)
        for summary in summaries:
            print(
                f"Polar disconnect={summary['polar_cross_plane_disconnect']}: "
                f"final peak-to-peak {summary['final_peak_to_peak_error_ns']:.3f} ns"
            )
    elif args.scenario == "node_failure_robustness":
        result = node_failure_robustness(config)
        print_summary(result["summary"])
    elif time_sync_config(config).get("monte_carlo", {}).get("enabled", False):
        summaries = run_monte_carlo(config)
        print(f"Completed {len(summaries)} Monte Carlo draws")
    else:
        run_dir = create_output_directory(config)
        result = run_single_simulation(config, run_dir)
        print_summary(result["summary"])


if __name__ == "__main__":
    main()
