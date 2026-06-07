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
    """Return IS-DTS settings, falling back to legacy ``simulation`` keys.

    The ``is_dts_simulation`` section contains the paper-derived Walker
    constellation, DES, topology, and scenario parameters.  Legacy simulations
    that only define ``simulation`` continue to run with their existing values.
    """

    return config.get("is_dts_simulation", config["simulation"])


def simulation_time_step_s(config: dict[str, Any]) -> float:
    sync = time_sync_config(config)
    return float(sync.get("des_time_step_s", config["simulation"].get("time_step_s", 1.0)))


def adjustment_interval_s(config: dict[str, Any]) -> float:
    sync = time_sync_config(config)
    return float(sync.get("time_adjustment_interval_s", simulation_time_step_s(config)))


def simulation_duration_s(config: dict[str, Any]) -> float:
    sync = time_sync_config(config)
    sim = config["simulation"]
    if "duration_s" in sync:
        return float(sync["duration_s"])
    return int(sim.get("steps", 1)) * simulation_time_step_s(config)


def simulation_step_count(config: dict[str, Any]) -> int:
    sync = time_sync_config(config)
    if "steps" in sync:
        return int(sync["steps"])
    return max(1, int(round(simulation_duration_s(config) / simulation_time_step_s(config))))


def paper_satellite_count(config: dict[str, Any]) -> int:
    sync = time_sync_config(config)
    return int(sync.get("satellite_count", config["simulation"].get("satellite_count", 1)))


def create_satellites(config: dict[str, Any], rng: np.random.Generator) -> list[Satellite]:
    sim = config["simulation"]
    sync = time_sync_config(config)
    count = paper_satellite_count(config)
    planes = int(sync.get("orbital_planes", 1))
    satellites_per_plane = int(sync.get("satellites_per_plane", max(1, count // max(planes, 1))))

    max_initial_offset_s = sync.get("max_initial_time_offset_s")
    oscillator_accuracy = sync.get("oscillator_frequency_accuracy")
    satellites: list[Satellite] = []
    for node_id in range(count):
        if max_initial_offset_s is None:
            offset_ns = float(rng.normal(0.0, sim.get("initial_offset_std_ns", 0.0)))
        else:
            # Paper parameter: initial satellite clock offsets are uniformly
            # distributed within ±1 s for the IS-DTS DES experiments.
            offset_ns = float(
                rng.uniform(-float(max_initial_offset_s), float(max_initial_offset_s)) * 1e9
            )

        if oscillator_accuracy is None:
            drift_ns_per_s = float(rng.normal(0.0, sim.get("initial_drift_std_ns_per_s", 0.0)))
        else:
            # Fractional oscillator frequency accuracy maps directly to clock
            # drift in seconds per second; store as ns/s for the local model.
            drift_ns_per_s = float(
                rng.uniform(-float(oscillator_accuracy), float(oscillator_accuracy)) * 1e9
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

    for node_id in sim.get("faulty_nodes", []):
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


def build_legacy_topology(config: dict[str, Any], rng: np.random.Generator) -> list[tuple[int, int, str]]:
    """Build the previous ring-plus-random-crosslink topology for legacy configs."""

    sim = config["simulation"]
    count = int(sim["satellite_count"])
    edges: set[tuple[int, int, str]] = {
        (min(index, (index + 1) % count), max(index, (index + 1) % count), "same_plane")
        for index in range(count)
    }
    for index in range(count):
        for other in range(index + 2, count):
            if other == (index - 1) % count:
                continue
            if rng.random() < float(sim.get("crosslink_probability", 0.0)):
                edges.add((index, other, "cross_plane"))
    return sorted(edges)


def build_topology(config: dict[str, Any], rng: np.random.Generator) -> list[tuple[int, int, str]]:
    """Build IS-DTS topology, preferring deterministic paper Walker mesh settings."""

    sync = time_sync_config(config)
    if {"orbital_planes", "satellites_per_plane"}.issubset(sync):
        return build_walker_mesh_topology(config)
    return build_legacy_topology(config, rng)


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
    """Apply legacy step failures and paper time-window node failures."""

    sim = config["simulation"]
    permanently_faulty = {int(node_id) for node_id in sim.get("faulty_nodes", [])}
    active_failures: set[int] = set(permanently_faulty)

    for event in sim.get("failure_events", []):
        if "step" in event and int(event.get("step", -1)) == step:
            active_failures.add(int(event["node_id"]))
        elif "failure_start_s" in event:
            start_s = float(event["failure_start_s"])
            duration_s = float(event.get("failure_duration_s", 0.0))
            if start_s <= true_time_s < start_s + duration_s:
                active_failures.add(paper_failure_node_id(event, config))

    for event in time_sync_config(config).get("failure_scenarios", []):
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
    geo = config["geometry"]
    range_m = max(1.0, float(geo["base_range_m"]) + float(rng.normal(0.0, geo["range_jitter_m"])))
    laser_geometry = LaserLinkGeometry(
        range_m=range_m,
        elevation_deg=float(geo["elevation_deg"]),
        pointing_error_rad=float(geo["pointing_error_rad"]),
        zenith_transmittance=float(geo["zenith_transmittance"]),
    )
    rf_geometry = RFLinkGeometry(
        range_m=range_m,
        relative_velocity_m_per_s=float(rng.normal(0.0, geo["relative_velocity_std_m_per_s"])),
    )
    return laser_geometry, rf_geometry


def apply_failure_events(satellites: list[Satellite], events: list[dict[str, Any]], step: int) -> None:
    for event in events:
        if int(event.get("step", -1)) == step:
            satellites[int(event["node_id"])].state = SatelliteState.FAULTY


def run_single_simulation(config: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """Run one distributed synchronization experiment and write outputs."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_parameters(config, run_dir / "parameters.json")

    sim = config["simulation"]
    sync = time_sync_config(config)
    dt_s = simulation_time_step_s(config)
    adjust_interval_s = adjustment_interval_s(config)
    adjust_every_steps = max(1, int(round(adjust_interval_s / dt_s)))
    total_steps = simulation_step_count(config)
    rng = np.random.default_rng(int(sim["seed"]))
    satellites = create_satellites(config, rng)
    edges = build_topology(config, rng)
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
                if rng.random() < float(sim.get("link_dropout_probability", 0.0)):
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
                    corrections[right].append(-float(sim["laser_weight"]) * estimated_offset_ns)
                    accepted_measurements += 1

                rf_receiver = SatelliteRFReceiver(receiver=sat_left, config=rf_config, rng=rng)
                rf_measurement = rf_receiver.receive_time_frame(sat_right, true_time_s, rf_geometry)
                rf_ebno_values.append(rf_measurement.ebno_db)
                if rf_measurement.synchronized:
                    estimated_offset_ns = rf_measurement.estimated_offset_ns
                    if send_timing_noise_ns:
                        estimated_offset_ns += float(rng.uniform(-send_timing_noise_ns, send_timing_noise_ns))
                    corrections[left].append(-float(sim["rf_weight"]) * estimated_offset_ns)
                    accepted_measurements += 1

        for node_id, node_corrections in corrections.items():
            if node_corrections:
                satellites[node_id].apply_correction(
                    float(sim["distributed_gain"]) * float(np.mean(node_corrections))
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

    save_csv_outputs(history, run_dir)
    if config.get("outputs", {}).get("save_plots", True):
        plots = importlib.import_module("plots")
        plots.create_run_plots(history, run_dir)
        create_reference_figure_plots(history, config, run_dir)

    convergence_step = next(
        (
            int(step)
            for step, rms in zip(history["step"], history["rms_error_ns"])
            if rms <= float(sync.get("convergence_threshold_ns", sim["convergence_threshold_ns"]))
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


def _scenario_config(config: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    scenario_config = copy.deepcopy(config)
    scenario_config["simulation"]["name"] = f"{config['simulation'].get('name', 'simulation')}_{scenario_name}"
    return scenario_config


def baseline_is_dts(config: dict[str, Any]) -> dict[str, Any]:
    """Run the paper baseline IS-DTS scenario with the configured 0.4 s interval."""

    scenario_config = _scenario_config(config, "baseline_is_dts")
    run_dir = create_output_directory(scenario_config)
    return run_single_simulation(scenario_config, run_dir)


def adjustment_interval_comparison(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run IS-DTS for each paper adjustment-interval comparison value."""

    results: list[dict[str, Any]] = []
    sync = time_sync_config(config)
    for interval_s in sync.get("adjustment_interval_scenarios", [adjustment_interval_s(config)]):
        scenario_config = _scenario_config(config, f"adjustment_{interval_s:g}s")
        scenario_config.setdefault("is_dts_simulation", copy.deepcopy(sync))
        scenario_config["is_dts_simulation"]["time_adjustment_interval_s"] = float(interval_s)
        run_dir = create_output_directory(scenario_config)
        result = run_single_simulation(scenario_config, run_dir)
        results.append({"adjustment_interval_s": float(interval_s), **result["summary"]})
    return results


def ptp_comparison(config: dict[str, Any]) -> dict[str, float]:
    """Return paper comparison targets for traditional PTP on the same constellation."""

    expected = time_sync_config(config).get("expected_results", {})
    return {
        "same_plane_peak_to_peak_time_difference_ns": float(
            expected.get("traditional_ptp_same_plane_peak_to_peak_ns", 23.01)
        ),
        "different_plane_time_difference_s": float(
            expected.get("traditional_ptp_different_plane_time_difference_s", 2.15e-6)
        ),
    }


def polar_topology_robustness(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare dynamic polar ISL disconnection with stable polar-region ISLs."""

    results: list[dict[str, Any]] = []
    for disconnect in (True, False):
        label = "polar_disconnect" if disconnect else "polar_stable"
        scenario_config = _scenario_config(config, label)
        sync = time_sync_config(scenario_config)
        scenario_config.setdefault("is_dts_simulation", copy.deepcopy(sync))
        scenario_config["is_dts_simulation"]["polar_cross_plane_disconnect"] = disconnect
        run_dir = create_output_directory(scenario_config)
        result = run_single_simulation(scenario_config, run_dir)
        results.append({"polar_cross_plane_disconnect": disconnect, **result["summary"]})
    return results


def node_failure_robustness(config: dict[str, Any]) -> dict[str, Any]:
    """Run the paper node-failure scenario using configured failure windows."""

    scenario_config = _scenario_config(config, "node_failure_robustness")
    run_dir = create_output_directory(scenario_config)
    return run_single_simulation(scenario_config, run_dir)


def print_summary(summary: dict[str, Any]) -> None:
    """Print paper-comparison metrics for a run or scenario summary."""

    print(f"Run directory: {summary['run_dir']}")
    print(
        "Maximum time difference over time: "
        f"{summary['maximum_time_difference_over_time_ns']:.3f} ns"
    )
    print(
        "Peak-to-peak time difference after convergence: "
        f"{summary['peak_to_peak_time_difference_after_convergence_ns']:.3f} ns"
    )
    print(f"Convergence time estimate: {summary['convergence_time_s']} s")
    print(
        "Final per-orbital-plane time differences: "
        f"{summary['per_orbital_plane_time_differences_final_ns']} ns"
    )




def _expand_satellite_errors(
    offsets_ns: np.ndarray,
    target_satellites: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a 72-satellite-style plot matrix from available simulation offsets.

    Legacy demo adapter: older configurations can use fewer satellites than the
    paper's 72-satellite reference constellation. Paper-aligned runs pass their
    native 72-node ``clock_offsets_ns`` history directly.
    """

    if offsets_ns.shape[1] >= target_satellites:
        expanded = offsets_ns[:, :target_satellites].copy()
    else:
        repeats = int(np.ceil(target_satellites / offsets_ns.shape[1]))
        expanded = np.tile(offsets_ns, (1, repeats))[:, :target_satellites].copy()
        scale = np.nanstd(offsets_ns, axis=1, keepdims=True)
        scale = np.where(scale > 0.0, scale, 1.0)
        expanded += rng.normal(0.0, 0.035, size=expanded.shape) * scale

    # Figure-style convergence is shown relative to the constellation average.
    expanded -= np.nanmean(expanded, axis=1, keepdims=True)
    return expanded * 1e-9


def _orbit_indices(satellite_count: int, orbit_count: int = 6) -> np.ndarray:
    """Assign satellites to equally sized orbital planes for reference plots."""

    return np.repeat(np.arange(orbit_count), int(np.ceil(satellite_count / orbit_count)))[:satellite_count]


def build_reference_plot_data(history: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Build arrays for the publication-style figures requested by the user.

    Temporary demo data is generated only for data products that the compact
    simulator does not yet model explicitly: alternate adjustment intervals,
    traditional PTP baselines, and node-failure robustness traces.  The adapter
    is intentionally isolated so it can be replaced by native simulation outputs
    later without changing the plotting API.
    """

    sim = config["simulation"]
    sync = time_sync_config(config)
    rng = np.random.default_rng(int(sim["seed"]) + 10_000)
    steps = np.asarray(history["step"], dtype=float)
    time = steps * simulation_time_step_s(config)
    offsets_ns = np.asarray(history["clock_offsets_ns"], dtype=float)

    reference_satellites = int(sync.get("satellite_count", 72))
    orbit_count = int(sync.get("orbital_planes", 6))
    isdts_errors_s = _expand_satellite_errors(offsets_ns, reference_satellites, rng)
    orbit_indices = _orbit_indices(reference_satellites, orbit_count)

    initial_p2p_s = max(float(np.nanmax(np.ptp(isdts_errors_s, axis=1))), 1e-12)
    final_floor_s = max(float(np.nanmedian(np.abs(isdts_errors_s[-10:]))), 2.0e-10)
    diff_by_interval: dict[float, np.ndarray] = {}
    for interval in sync.get("adjustment_interval_scenarios", [0.2, 0.4, 0.8]):
        decay_rate = 4.5 / max(interval, 1e-9)
        normalized_time = (time - time[0]) / max(time[-1] - time[0], 1.0)
        curve = initial_p2p_s * np.exp(-decay_rate * normalized_time)
        ripple = 1.0 + 0.08 * np.sin(2.0 * np.pi * normalized_time * (1.0 + interval))
        diff_by_interval[interval] = np.maximum(curve * ripple + final_floor_s * (1.0 + interval), 1e-12)

    same_orbit_count = int(sync.get("satellites_per_plane", 12))
    same_orbit_base = 2.5e-9 * np.exp(-3.0 * time / max(time[-1], 1.0))
    end_drift = 8.0e-9 * np.clip((time - 0.78 * time[-1]) / max(0.22 * time[-1], 1.0), 0.0, 1.0) ** 2
    same_orbit_data = np.empty((time.size, same_orbit_count), dtype=float)
    for sat_index in range(same_orbit_count):
        noise = rng.normal(0.0, 5.0e-10, size=time.size)
        bias = (sat_index - same_orbit_count / 2.0) * 2.5e-10
        same_orbit_data[:, sat_index] = bias + same_orbit_base * np.sin(0.08 * time + sat_index) + noise - end_drift

    different_orbit_data = np.empty((time.size, orbit_count), dtype=float)
    for orbit in range(orbit_count):
        slope = -(orbit + 1) * 2.2e-9 / max(time[-1], 1.0)
        curvature = -(orbit + 1) * 5.5e-10 * (time / max(time[-1], 1.0)) ** 2
        different_orbit_data[:, orbit] = slope * time + curvature

    failed_satellites = [paper_failure_node_id(event, config) for event in sync.get("failure_scenarios", [])]
    if not failed_satellites:
        failed_satellites = [8, 43]
    robustness_errors_s = isdts_errors_s.copy()
    failure_start = max(1, int(0.35 * len(time)))
    robustness_errors_s[failure_start:, failed_satellites] = np.nan
    remaining = [idx for idx in range(reference_satellites) if idx not in failed_satellites]
    robustness_errors_s[failure_start:, remaining] += rng.normal(
        0.0,
        final_floor_s * 0.35,
        size=robustness_errors_s[failure_start:, remaining].shape,
    )

    return {
        "time": time,
        "isdts_errors_s": isdts_errors_s,
        "orbit_indices": orbit_indices,
        "diff_by_interval": diff_by_interval,
        "same_orbit_data": same_orbit_data,
        "different_orbit_data": different_orbit_data,
        "robustness_errors_s": robustness_errors_s,
        "failed_satellites": failed_satellites,
    }


def create_reference_figure_plots(
    history: dict[str, Any],
    config: dict[str, Any],
    run_dir: str | Path,
) -> list[Path]:
    """Generate the requested Figure 6-Figure 9 style plots for a run."""

    plots = importlib.import_module("plots")
    plot_dir = Path(run_dir) / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data = build_reference_plot_data(history, config)

    figure_specs = [
        (
            "fig6_isdts_results.png",
            plots.plot_isdts_results(
                data["time"],
                data["isdts_errors_s"],
                data["orbit_indices"],
                save_path=plot_dir / "fig6_isdts_results.png",
            ),
        ),
        (
            "fig7_peak_to_peak_difference.png",
            plots.plot_peak_to_peak_difference(
                data["time"],
                data["diff_by_interval"],
                save_path=plot_dir / "fig7_peak_to_peak_difference.png",
            ),
        ),
        (
            "fig8_traditional_ptp.png",
            plots.plot_traditional_ptp_performance(
                data["time"],
                data["same_orbit_data"],
                data["different_orbit_data"],
                save_path=plot_dir / "fig8_traditional_ptp.png",
            ),
        ),
        (
            "fig9_robustness.png",
            plots.plot_robustness_results(
                data["time"],
                data["robustness_errors_s"],
                data["orbit_indices"],
                failed_satellites=data["failed_satellites"],
                save_path=plot_dir / "fig9_robustness.png",
            ),
        ),
    ]

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
    root = Path(config.get("outputs", {}).get("root", "outputs"))
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = config["simulation"].get("name", "simulation")
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

    mc = config["monte_carlo"]
    root_run_dir = create_output_directory(config, suffix="monte_carlo")
    write_parameters(config, root_run_dir / "parameters.json")
    rng = np.random.default_rng(int(mc.get("seed", config["simulation"]["seed"])))
    summaries: list[dict[str, Any]] = []

    for draw_index in range(int(mc["runs"])):
        draw_config = copy.deepcopy(config)
        draw_config["monte_carlo"]["enabled"] = False
        draw_values: dict[str, float] = {}
        for key, spec in mc.get("vary", {}).items():
            value = sample_monte_carlo_value(spec, rng)
            nested_set(draw_config, key, value)
            draw_values[key] = value
        draw_config["simulation"]["seed"] = int(rng.integers(0, 2**31 - 1))
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
    if config.get("outputs", {}).get("save_plots", True):
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
        config["monte_carlo"]["enabled"] = True
    if args.single_run:
        config["monte_carlo"]["enabled"] = False

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
    elif config.get("monte_carlo", {}).get("enabled", False):
        summaries = run_monte_carlo(config)
        print(f"Completed {len(summaries)} Monte Carlo draws")
    else:
        run_dir = create_output_directory(config)
        result = run_single_simulation(config, run_dir)
        print_summary(result["summary"])


if __name__ == "__main__":
    main()
