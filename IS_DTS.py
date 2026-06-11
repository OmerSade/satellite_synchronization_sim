"""Paper-aligned IS-DTS, RF receiver, and laser communication simulation runner.

This self-contained runner implements a deterministic discrete-event simulation
for the paper "Inter-satellite distributed time synchronization solution with
nanosecond accuracy in satellite networks" while reusing the repository's RF
receiver, laser communication, and plotting modules.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import types
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
DEFAULT_PARAMETER_CANDIDATES = ("parameters", "parameters.json", "parameters.py")


class SatelliteState(Enum):
    """State machine shared by the IS-DTS runner and communication modules."""

    INITIALIZING = "Initializing"
    LISTENING = "Listening"
    ADJUSTING = "Adjusting"
    FAULTY = "Faulty"


@dataclass
class SatelliteNode:
    """One satellite clock, orbit, state, and message-buffer record."""

    node_id: int
    orbital_plane: int
    satellite_index: int
    clock_offset_ns: float
    drift_ns_per_s: float
    position_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity_m_per_s: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    latitude_deg: float = 0.0
    state: SatelliteState = SatelliteState.INITIALIZING
    isdts_buffer: list[dict[str, Any]] = field(default_factory=list)
    ptp_buffer: list[dict[str, Any]] = field(default_factory=list)
    history: dict[str, list[float]] = field(default_factory=dict)

    @property
    def offset_ns(self) -> float:
        """Compatibility alias used by satellite_receiver.py and laser module."""

        return self.clock_offset_ns

    @offset_ns.setter
    def offset_ns(self, value: float) -> None:
        self.clock_offset_ns = float(value)

    @property
    def is_healthy(self) -> bool:
        """Return True if this node can participate in synchronization."""

        return self.state != SatelliteState.FAULTY

    def complete_initialization(self) -> None:
        """Move a healthy node from Initializing to Listening."""

        if self.state == SatelliteState.INITIALIZING:
            self.state = SatelliteState.LISTENING

    def tick(self, dt_s: float) -> None:
        """Advance the free-running local oscillator by one DES interval."""

        if self.is_healthy:
            self.clock_offset_ns += self.drift_ns_per_s * dt_s

    def local_time_ns(self, true_time_s: float) -> float:
        """Return this node's local clock reading in nanoseconds."""

        return true_time_s * 1e9 + self.clock_offset_ns

    def apply_correction(self, correction_ns: float) -> None:
        """Apply a clock correction without changing faulty nodes."""

        if self.is_healthy:
            self.state = SatelliteState.ADJUSTING
            self.clock_offset_ns += float(correction_ns)


def _install_satellite_compatibility_module() -> None:
    """Expose local compatibility types as src.satellite when needed."""

    if "src.satellite" in sys.modules:
        return
    src_module = sys.modules.setdefault("src", types.ModuleType("src"))
    satellite_module = types.ModuleType("src.satellite")
    satellite_module.Satellite = SatelliteNode
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


@dataclass(frozen=True)
class LinkEdge:
    """Undirected inter-satellite link edge."""

    left: int
    right: int
    edge_type: str


@dataclass
class RunProducts:
    """Container returned by a simulation run."""

    run_dir: Path
    history: dict[str, Any]
    summary: dict[str, Any]
    generated_plots: list[Path]
    parameters_added: bool


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dataclass_from_dict(cls: type[Any], data: dict[str, Any]) -> Any:
    """Construct a dataclass using only supported field names."""

    valid_names = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in valid_names})


def load_parameters(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    """Load JSON parameters, with parameters.py as a compatibility fallback."""

    candidates = [Path(path)] if path else [Path(name) for name in DEFAULT_PARAMETER_CANDIDATES]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".py":
            spec = importlib.util.spec_from_file_location("parameters_module", candidate)
            if spec is None or spec.loader is None:
                raise ValueError(f"Unable to import parameter module: {candidate}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "PARAMETERS"):
                return dict(module.PARAMETERS), candidate
            if hasattr(module, "parameters"):
                return dict(module.parameters), candidate
            raise ValueError(f"{candidate} must define PARAMETERS or parameters")
        with candidate.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj), candidate
    raise FileNotFoundError("No parameters file found; tried parameters, parameters.json, parameters.py")


def write_parameters(config: dict[str, Any], path: str | Path) -> None:
    """Write parameters in readable JSON format."""

    with Path(path).open("w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2, sort_keys=False, default=_json_default)
        file_obj.write("\n")


def ensure_parameter_defaults(config: dict[str, Any], parameter_path: Path) -> bool:
    """Add missing paper-required derived/default parameters to the root JSON file."""

    if "is_dts_simulation" not in config:
        raise ValueError("Missing required top-level section: is_dts_simulation")
    sync = config["is_dts_simulation"]
    planes = int(sync.get("orbital_planes", 0) or 0)
    sats_per_plane = int(sync.get("satellites_per_plane", 0) or 0)
    defaults = {
        "earth_radius_m": 6_371_000.0,
        "earth_mu_m3_s2": 3.986004418e14,
        "orbital_period_s": None,
        "raan_spacing_deg": 360.0 / planes if planes else None,
        "true_anomaly_spacing_deg": 360.0 / sats_per_plane if sats_per_plane else None,
        "plane_phase_offset_deg": 0.0,
        "ptp_master_node_id": 0,
        "ptp_exchange_interval_s": sync.get("time_adjustment_interval_s"),
        "output_root": "outputs",
        "metrics_csv_name": "metrics.csv",
        "offsets_csv_name": "clock_offsets.csv",
        "links_csv_name": "link_measurements.csv",
        "summary_json_name": "summary.json",
    }
    added = False
    for key, value in defaults.items():
        if key not in sync:
            sync[key] = value
            added = True
    if parameter_path.suffix != ".py" and added:
        write_parameters(config, parameter_path)
    return added


def validate_parameters(config: dict[str, Any]) -> None:
    """Validate required paper, RF, laser, and output parameters."""

    for section in ("is_dts_simulation", "rf", "laser"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing required parameter section: {section}")
    sync = config["is_dts_simulation"]
    required = (
        "satellite_count",
        "orbital_planes",
        "satellites_per_plane",
        "altitude_m",
        "inclination_deg",
        "polar_disconnect_latitude_deg",
        "des_time_step_s",
        "time_adjustment_interval_s",
        "max_initial_time_offset_s",
        "oscillator_frequency_accuracy",
        "timing_difference_sending_messages_s",
        "same_plane_neighbors",
        "cross_plane_neighbors",
        "failure_scenarios",
        "adjustment_interval_scenarios",
        "link_geometry",
        "expected_results",
    )
    missing = [key for key in required if key not in sync]
    if missing:
        raise ValueError(f"Missing is_dts_simulation parameter(s): {', '.join(missing)}")
    if int(sync["satellite_count"]) != int(sync["orbital_planes"]) * int(sync["satellites_per_plane"]):
        raise ValueError("satellite_count must equal orbital_planes * satellites_per_plane")
    if int(sync["same_plane_neighbors"]) != 2:
        raise ValueError("same_plane_neighbors must be 2 for the paper-aligned simulation")
    if int(sync["cross_plane_neighbors"]) != 2:
        raise ValueError("cross_plane_neighbors must be 2 for the paper-aligned simulation")
    if float(sync["des_time_step_s"]) <= 0.0:
        raise ValueError("des_time_step_s must be > 0")
    if float(sync["time_adjustment_interval_s"]) < float(sync["des_time_step_s"]):
        raise ValueError("time_adjustment_interval_s must be >= des_time_step_s")
    if float(sync["altitude_m"]) <= 0.0:
        raise ValueError("altitude_m must be > 0")
    if not 0.0 <= float(sync["inclination_deg"]) <= 180.0:
        raise ValueError("inclination_deg must be in [0, 180]")
    Path(sync.get("output_root", "outputs")).mkdir(parents=True, exist_ok=True)


def orbital_period_s(sync: dict[str, Any]) -> float:
    """Return configured or circular-orbit period in seconds."""

    if sync.get("orbital_period_s") is not None:
        return float(sync["orbital_period_s"])
    radius = float(sync["earth_radius_m"]) + float(sync["altitude_m"])
    return float(2.0 * math.pi * math.sqrt(radius**3 / float(sync["earth_mu_m3_s2"])))


def _rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def propagate_node(node: SatelliteNode, time_s: float, sync: dict[str, Any]) -> None:
    """Propagate a deterministic circular Walker-like orbit to one time."""

    radius = float(sync["earth_radius_m"]) + float(sync["altitude_m"])
    mu = float(sync["earth_mu_m3_s2"])
    mean_motion = math.sqrt(mu / radius**3)
    raan = math.radians(float(sync["raan_spacing_deg"]) * node.orbital_plane)
    inclination = math.radians(float(sync["inclination_deg"]))
    phase_offset = math.radians(float(sync.get("plane_phase_offset_deg", 0.0)) * node.orbital_plane)
    anomaly0 = math.radians(float(sync["true_anomaly_spacing_deg"]) * node.satellite_index) + phase_offset
    theta = anomaly0 + mean_motion * time_s
    position_orbital = np.array([radius * math.cos(theta), radius * math.sin(theta), 0.0], dtype=float)
    velocity_orbital = np.array(
        [-radius * mean_motion * math.sin(theta), radius * mean_motion * math.cos(theta), 0.0],
        dtype=float,
    )
    transform = _rotation_matrix_z(raan) @ _rotation_matrix_x(inclination)
    node.position_m = transform @ position_orbital
    node.velocity_m_per_s = transform @ velocity_orbital
    rho = float(np.linalg.norm(node.position_m))
    node.latitude_deg = math.degrees(math.asin(float(node.position_m[2]) / rho))


def create_nodes(config: dict[str, Any], rng: np.random.Generator) -> list[SatelliteNode]:
    """Create reproducible satellite nodes with paper-specified clocks."""

    sync = config["is_dts_simulation"]
    nodes = []
    max_offset_s = float(sync["max_initial_time_offset_s"])
    oscillator_accuracy = float(sync["oscillator_frequency_accuracy"])
    sats_per_plane = int(sync["satellites_per_plane"])
    for node_id in range(int(sync["satellite_count"])):
        nodes.append(
            SatelliteNode(
                node_id=node_id,
                orbital_plane=node_id // sats_per_plane,
                satellite_index=node_id % sats_per_plane,
                clock_offset_ns=float(rng.uniform(-max_offset_s, max_offset_s) * 1e9),
                drift_ns_per_s=float(rng.uniform(-oscillator_accuracy, oscillator_accuracy) * 1e9),
            )
        )
    for node in nodes:
        node.complete_initialization()
        propagate_node(node, 0.0, sync)
    return nodes


def clone_nodes(nodes: Iterable[SatelliteNode]) -> list[SatelliteNode]:
    """Deep-copy the state needed to run IS-DTS and PTP baselines separately."""

    copied: list[SatelliteNode] = []
    for node in nodes:
        copied.append(
            SatelliteNode(
                node_id=node.node_id,
                orbital_plane=node.orbital_plane,
                satellite_index=node.satellite_index,
                clock_offset_ns=node.clock_offset_ns,
                drift_ns_per_s=node.drift_ns_per_s,
                position_m=np.array(node.position_m, dtype=float),
                velocity_m_per_s=np.array(node.velocity_m_per_s, dtype=float),
                latitude_deg=node.latitude_deg,
                state=node.state,
            )
        )
    return copied


def node_id_for(plane: int, sat_index: int, satellites_per_plane: int) -> int:
    """Return the deterministic node ID for a Walker plane/index pair."""

    return plane * satellites_per_plane + sat_index


def build_topology(config: dict[str, Any]) -> list[LinkEdge]:
    """Build same-plane and same-direction adjacent cross-plane ISLs."""

    sync = config["is_dts_simulation"]
    planes = int(sync["orbital_planes"])
    sats_per_plane = int(sync["satellites_per_plane"])
    edges: set[tuple[int, int, str]] = set()
    for plane in range(planes):
        for sat_index in range(sats_per_plane):
            node = node_id_for(plane, sat_index, sats_per_plane)
            ahead = node_id_for(plane, (sat_index + 1) % sats_per_plane, sats_per_plane)
            edges.add((min(node, ahead), max(node, ahead), "same_plane"))
            # Adjacent Walker planes are treated as the same-direction cross-plane
            # neighbors used by the paper; opposite-direction links are not added.
            for neighbor_plane in ((plane - 1) % planes, (plane + 1) % planes):
                other = node_id_for(neighbor_plane, sat_index, sats_per_plane)
                edges.add((min(node, other), max(node, other), "cross_plane"))
    return [LinkEdge(*edge) for edge in sorted(edges)]


def _failure_node_ids(sync: dict[str, Any], time_s: float) -> set[int]:
    failed: set[int] = set()
    planes = int(sync["orbital_planes"])
    sats_per_plane = int(sync["satellites_per_plane"])
    for failure in sync.get("failure_scenarios", []):
        start = float(failure.get("failure_start_s", failure.get("start_s", 0.0)))
        duration = float(failure.get("failure_duration_s", failure.get("duration_s", 0.0)))
        if not (start <= time_s < start + duration):
            continue
        if "node_id" in failure:
            node_id = int(failure["node_id"])
        else:
            plane_raw = int(failure.get("orbital_plane", 0))
            sat_raw = int(failure.get("satellite_index", 0))
            plane = plane_raw - 1 if 1 <= plane_raw <= planes else plane_raw
            sat_index = sat_raw - 1 if 1 <= sat_raw <= sats_per_plane else sat_raw
            node_id = node_id_for(plane % planes, sat_index % sats_per_plane, sats_per_plane)
        failed.add(node_id)
    return failed


def update_node_states(nodes: list[SatelliteNode], edges: list[LinkEdge], sync: dict[str, Any], time_s: float) -> None:
    """Apply initialization, failure windows, and Listening/Adjusting decisions."""

    failed = _failure_node_ids(sync, time_s)
    active_degree = {node.node_id: 0 for node in nodes}
    for edge in edges:
        if edge_is_active(edge, nodes, sync, ignore_dropout=True):
            active_degree[edge.left] += 1
            active_degree[edge.right] += 1
    for node in nodes:
        if node.node_id in failed:
            node.state = SatelliteState.FAULTY
        elif node.state == SatelliteState.FAULTY:
            node.state = SatelliteState.LISTENING
        elif node.state == SatelliteState.INITIALIZING:
            node.state = SatelliteState.LISTENING
        elif active_degree[node.node_id] > 0:
            node.state = SatelliteState.ADJUSTING
        else:
            node.state = SatelliteState.LISTENING


def edge_is_active(edge: LinkEdge, nodes: list[SatelliteNode], sync: dict[str, Any], *, ignore_dropout: bool = False, rng: np.random.Generator | None = None) -> bool:
    """Return whether an edge passes health, polar, and optional dropout gates."""

    left = nodes[edge.left]
    right = nodes[edge.right]
    if not left.is_healthy or not right.is_healthy:
        return False
    if edge.edge_type == "cross_plane" and bool(sync.get("polar_cross_plane_disconnect", True)):
        limit = float(sync["polar_disconnect_latitude_deg"])
        if abs(left.latitude_deg) > limit or abs(right.latitude_deg) > limit:
            return False
    if not ignore_dropout and rng is not None and rng.random() < float(sync.get("link_dropout_probability", 0.0)):
        return False
    return True


def link_geometry(left: SatelliteNode, right: SatelliteNode, sync: dict[str, Any], rng: np.random.Generator) -> tuple[float, float, RFLinkGeometry, LaserLinkGeometry]:
    """Build RF and laser geometry from current satellite positions/velocities."""

    vector = right.position_m - left.position_m
    range_m = max(float(np.linalg.norm(vector)), 1.0)
    los = vector / range_m
    relative_velocity = float(np.dot(right.velocity_m_per_s - left.velocity_m_per_s, los))
    geometry_cfg = sync.get("link_geometry", {})
    pointing_error = abs(float(geometry_cfg.get("pointing_error_rad", 1.0e-6)))
    if "pointing_error_jitter_rad" in geometry_cfg:
        pointing_error = abs(float(rng.normal(pointing_error, float(geometry_cfg["pointing_error_jitter_rad"]))))
    laser_geometry = LaserLinkGeometry(
        range_m=range_m,
        elevation_deg=float(geometry_cfg.get("elevation_deg", 60.0)),
        pointing_error_rad=pointing_error,
        zenith_transmittance=float(geometry_cfg.get("zenith_transmittance", 0.9)),
    )
    rf_geometry = RFLinkGeometry(range_m=range_m, relative_velocity_m_per_s=relative_velocity)
    return range_m, relative_velocity, rf_geometry, laser_geometry


def isdts_pairwise_offset_ns(t1_i_ns: float, t2_j_ns: float, t1_j_ns: float, t2_i_ns: float, delta_v_m_per_s: float) -> float:
    """Paper timestamp equation: neighbor j offset relative to node i, in ns."""

    t1 = t1_i_ns / 1e9
    t2 = t2_j_ns / 1e9
    t1_prime = t1_j_ns / 1e9
    t2_prime = t2_i_ns / 1e9
    offset_s = 0.5 * (t2 - t2_prime) + 0.5 * (t1_prime - t1) + (delta_v_m_per_s * (t1_prime - t1)) / (2.0 * SPEED_OF_LIGHT_M_PER_S)
    return float(offset_s * 1e9)


def ptp_offset_delay_ns(t1_ns: float, t2_ns: float, t3_ns: float, t4_ns: float) -> tuple[float, float]:
    """Standard PTP offset and delay equations in nanoseconds."""

    offset = ((t2_ns - t1_ns) - (t4_ns - t3_ns)) / 2.0
    delay = ((t2_ns - t1_ns) + (t4_ns - t3_ns)) / 2.0
    return float(offset), float(delay)


def exchange_isdts(
    edge: LinkEdge,
    nodes: list[SatelliteNode],
    time_s: float,
    step: int,
    sync: dict[str, Any],
    rf_config: RFReceiverConfig,
    laser_config: LaserLinkConfig,
    rng: np.random.Generator,
) -> tuple[dict[int, list[float]], dict[str, Any]]:
    """Run the near-parallel IS-DTS message exchange over one active ISL."""

    left = nodes[edge.left]
    right = nodes[edge.right]
    range_m, relative_velocity, rf_geometry, laser_geometry = link_geometry(left, right, sync, rng)
    mismatch_s = float(rng.uniform(-float(sync["timing_difference_sending_messages_s"]), float(sync["timing_difference_sending_messages_s"])))
    delay_lr_s = range_m / SPEED_OF_LIGHT_M_PER_S
    delay_rl_s = max(0.0, (range_m + relative_velocity * mismatch_s) / SPEED_OF_LIGHT_M_PER_S)

    rf_lr = SatelliteRFReceiver(receiver=right, config=rf_config, rng=rng).receive_time_frame(left, time_s, rf_geometry)
    laser_lr = SatelliteLaserCommunication(receiver=right, config=laser_config, rng=rng).measure_downlink(left, time_s, laser_geometry)

    t1_left = left.local_time_ns(time_s)
    t2_right = right.local_time_ns(time_s + delay_lr_s)
    t1_right = right.local_time_ns(time_s + mismatch_s)
    t2_left = left.local_time_ns(time_s + mismatch_s + delay_rl_s)
    base_offset_right_relative_left = isdts_pairwise_offset_ns(t1_left, t2_right, t1_right, t2_left, relative_velocity)
    noise_ns = weighted_measurement_noise_ns(rf_lr, laser_lr, sync=sync)
    offset_right_relative_left = base_offset_right_relative_left + noise_ns
    valid = bool(rf_lr.synchronized and laser_lr.acquired)
    corrections: dict[int, list[float]] = {left.node_id: [], right.node_id: []}
    if valid:
        corrections[left.node_id].append(offset_right_relative_left)
        corrections[right.node_id].append(-offset_right_relative_left)
    record = link_record(step, time_s, "isdts", edge, range_m, relative_velocity, rf_lr, laser_lr, offset_right_relative_left, valid, valid)
    return corrections, record


def weighted_measurement_noise_ns(*measurements: Any, sync: dict[str, Any]) -> float:
    """Use RF/laser timestamp estimates only as small link-quality noise inputs."""

    values = [float(getattr(item, "estimated_offset_ns", 0.0)) for item in measurements]
    if not values:
        return 0.0
    # Common-mode offsets are dominated by the explicit timestamp equation; the
    # residual term keeps communication-module quality coupled to the estimate.
    return 0.02 * float(np.std(values)) * float(sync.get("rf_weight", 0.25) + sync.get("laser_weight", 0.75))


def link_record(
    step: int,
    time_s: float,
    method: str,
    edge: LinkEdge,
    range_m: float,
    relative_velocity: float,
    rf_measurement: Any,
    laser_measurement: Any,
    estimated_offset_ns: float,
    used: bool,
    valid: bool,
) -> dict[str, Any]:
    """Create one CSV-ready link-measurement row."""

    return {
        "step": step,
        "time_s": time_s,
        "method": method,
        "left_satellite": edge.left,
        "right_satellite": edge.right,
        "edge_type": edge.edge_type,
        "range_m": range_m,
        "relative_velocity_m_per_s": relative_velocity,
        "rf_valid": bool(getattr(rf_measurement, "synchronized", False)),
        "rf_ebno_db": float(getattr(rf_measurement, "ebno_db", math.nan)),
        "rf_ber": float(getattr(rf_measurement, "ber", math.nan)),
        "rf_received_power_dbw": float(getattr(rf_measurement, "received_power_dbw", math.nan)),
        "laser_valid": bool(getattr(laser_measurement, "acquired", False)),
        "laser_snr_db": float(getattr(laser_measurement, "snr_db", math.nan)),
        "laser_received_power_dbw": float(getattr(laser_measurement, "received_power_dbw", math.nan)),
        "laser_photons_per_bit": float(getattr(laser_measurement, "photons_per_bit", math.nan)),
        "laser_capacity_bps": float(getattr(laser_measurement, "capacity_bps", math.nan)),
        "estimated_offset_ns": float(estimated_offset_ns),
        "used_for_correction": bool(used and valid),
    }


def adjacency_from_edges(edges: list[LinkEdge], nodes: list[SatelliteNode], sync: dict[str, Any]) -> dict[int, list[int]]:
    """Build active-edge adjacency for shortest-path PTP synchronization."""

    adjacency = {node.node_id: [] for node in nodes if node.is_healthy}
    for edge in edges:
        if edge_is_active(edge, nodes, sync, ignore_dropout=True):
            adjacency.setdefault(edge.left, []).append(edge.right)
            adjacency.setdefault(edge.right, []).append(edge.left)
    return adjacency


def shortest_paths_from_master(adjacency: dict[int, list[int]], master: int) -> dict[int, int]:
    """Return predecessor toward master for each reachable node."""

    predecessor: dict[int, int] = {}
    visited = {master}
    queue: deque[int] = deque([master])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            predecessor[neighbor] = current
            queue.append(neighbor)
    return predecessor


def exchange_ptp(
    master_like: SatelliteNode,
    slave: SatelliteNode,
    edge_type: str,
    time_s: float,
    step: int,
    sync: dict[str, Any],
    rf_config: RFReceiverConfig,
    laser_config: LaserLinkConfig,
    rng: np.random.Generator,
) -> tuple[float | None, dict[str, Any]]:
    """Run a serial PTP Sync/Delay_Req/Delay_Resp exchange for one hop."""

    edge = LinkEdge(min(master_like.node_id, slave.node_id), max(master_like.node_id, slave.node_id), edge_type)
    range_m, relative_velocity, rf_geometry, laser_geometry = link_geometry(master_like, slave, sync, rng)
    exchange_gap_s = max(
        float(sync["timing_difference_sending_messages_s"]),
        float(sync.get("ptp_exchange_interval_s", sync["time_adjustment_interval_s"])),
    )
    delay1_s = range_m / SPEED_OF_LIGHT_M_PER_S
    range2_m = max(1.0, range_m + relative_velocity * exchange_gap_s)
    delay2_s = range2_m / SPEED_OF_LIGHT_M_PER_S
    t1 = master_like.local_time_ns(time_s)
    t2 = slave.local_time_ns(time_s + delay1_s)
    t3 = slave.local_time_ns(time_s + delay1_s + exchange_gap_s)
    t4 = master_like.local_time_ns(time_s + delay1_s + exchange_gap_s + delay2_s)
    offset_ns, _delay_ns = ptp_offset_delay_ns(t1, t2, t3, t4)
    rf_measurement = SatelliteRFReceiver(receiver=slave, config=rf_config, rng=rng).receive_time_frame(master_like, time_s, rf_geometry)
    laser_measurement = SatelliteLaserCommunication(receiver=slave, config=laser_config, rng=rng).measure_downlink(master_like, time_s, laser_geometry)
    valid = bool(rf_measurement.synchronized and laser_measurement.acquired)
    used = valid
    correction = -offset_ns if valid else None
    record = link_record(step, time_s, "ptp", edge, range_m, relative_velocity, rf_measurement, laser_measurement, offset_ns, used, valid)
    return correction, record


def compute_error_metrics(nodes: list[SatelliteNode]) -> tuple[float, float]:
    """Return peak-to-peak and RMS error relative to healthy-node mean."""

    offsets = np.asarray([node.clock_offset_ns for node in nodes if node.is_healthy], dtype=float)
    if offsets.size == 0:
        return 0.0, 0.0
    centered = offsets - float(np.mean(offsets))
    return float(np.ptp(offsets)), float(math.sqrt(float(np.mean(centered**2))))


def run_sanity_checks(config: dict[str, Any]) -> None:
    """Run startup sanity checks without requiring pytest."""

    rng = np.random.default_rng(int(config["is_dts_simulation"].get("seed", 0)))
    nodes = create_nodes(config, rng)
    edges = build_topology(config)
    same = [edge for edge in edges if edge.edge_type == "same_plane"]
    cross = [edge for edge in edges if edge.edge_type == "cross_plane"]
    if not same or not cross:
        raise ValueError("Sanity check failed: topology must contain same-plane and cross-plane links")
    max_offset_ns = float(config["is_dts_simulation"]["max_initial_time_offset_s"]) * 1e9
    if any(abs(node.clock_offset_ns) > max_offset_ns for node in nodes):
        raise ValueError("Sanity check failed: clocks are outside ±max_initial_time_offset_s")
    rf_config = dataclass_from_dict(RFReceiverConfig, config["rf"])
    laser_config = dataclass_from_dict(LaserLinkConfig, config["laser"])
    first_edge = edges[0]
    range_m, relative_velocity, rf_geometry, laser_geometry = link_geometry(nodes[first_edge.left], nodes[first_edge.right], config["is_dts_simulation"], rng)
    rf = SatelliteRFReceiver(receiver=nodes[first_edge.right], config=rf_config, rng=rng).receive_time_frame(nodes[first_edge.left], 0.0, rf_geometry)
    if not hasattr(rf, "ebno_db") or not hasattr(rf, "synchronized"):
        raise ValueError("Sanity check failed: RF measurement lacks ebno_db or synchronized")
    laser = SatelliteLaserCommunication(receiver=nodes[first_edge.right], config=laser_config, rng=rng).measure_downlink(nodes[first_edge.left], 0.0, laser_geometry)
    if not hasattr(laser, "snr_db") or not hasattr(laser, "acquired"):
        raise ValueError("Sanity check failed: laser measurement lacks snr_db or acquired")
    if not math.isfinite(isdts_pairwise_offset_ns(0.0, 10.0, 1.0, 11.0, relative_velocity)):
        raise ValueError("Sanity check failed: IS-DTS pairwise offset is not finite")
    offset, delay = ptp_offset_delay_ns(0.0, range_m / SPEED_OF_LIGHT_M_PER_S * 1e9, 1000.0, 1000.0 + range_m / SPEED_OF_LIGHT_M_PER_S * 1e9)
    if not math.isfinite(offset) or not math.isfinite(delay):
        raise ValueError("Sanity check failed: PTP offset/delay is not finite")


def run_simulation(config: dict[str, Any], parameter_path: Path, method: Literal["both", "isdts", "ptp"], no_plots: bool, parameters_added: bool) -> RunProducts:
    """Execute the DES loop, save CSV/JSON outputs, and create plots."""

    sync = config["is_dts_simulation"]
    rng = np.random.default_rng(int(sync.get("seed", 0)))
    initial_nodes = create_nodes(config, rng)
    isdts_nodes = clone_nodes(initial_nodes)
    ptp_nodes = clone_nodes(initial_nodes)
    edges = build_topology(config)
    rf_config = dataclass_from_dict(RFReceiverConfig, config["rf"])
    laser_config = dataclass_from_dict(LaserLinkConfig, config["laser"])
    dt_s = float(sync.get("des_time_step_s", 0.02))
    duration_s = float(sync.get("duration_s", 20.0))
    steps = int(round(duration_s / dt_s))
    adjust_every = max(1, int(round(float(sync["time_adjustment_interval_s"]) / dt_s)))
    ptp_every = max(1, int(round(float(sync.get("ptp_exchange_interval_s", sync["time_adjustment_interval_s"])) / dt_s)))
    run_dir = Path(sync.get("output_root", "outputs")) / f"{sync.get('name', 'is_dts')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    metrics_rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    isdts_clock_history: list[list[float]] = []
    ptp_clock_history: list[list[float]] = []
    total_isdts = 0
    total_ptp = 0

    for step in range(steps + 1):
        time_s = step * dt_s
        for nodes in (isdts_nodes, ptp_nodes):
            for node in nodes:
                propagate_node(node, time_s, sync)
                if step > 0:
                    node.tick(dt_s)
            update_node_states(nodes, edges, sync, time_s)

        active_edges = [edge for edge in edges if edge_is_active(edge, isdts_nodes, sync, rng=rng)]
        active_same = sum(1 for edge in active_edges if edge.edge_type == "same_plane")
        active_cross = sum(1 for edge in active_edges if edge.edge_type == "cross_plane")
        accepted_isdts = 0
        accepted_ptp = 0
        rf_values: list[float] = []
        laser_values: list[float] = []

        if method in ("both", "isdts") and step % adjust_every == 0:
            corrections: dict[int, list[float]] = {node.node_id: [] for node in isdts_nodes if node.is_healthy}
            for edge in active_edges:
                edge_corrections, record = exchange_isdts(edge, isdts_nodes, time_s, step, sync, rf_config, laser_config, rng)
                link_rows.append(record)
                rf_values.append(record["rf_ebno_db"])
                laser_values.append(record["laser_snr_db"])
                if record["used_for_correction"]:
                    accepted_isdts += 1
                    for node_id, values in edge_corrections.items():
                        corrections.setdefault(node_id, []).extend(values)
            for node_id, values in corrections.items():
                if values:
                    correction = float(sync.get("distributed_gain", 0.65)) * (float(np.sum(values)) / (len(values) + 1))
                    isdts_nodes[node_id].apply_correction(correction)
            for node in isdts_nodes:
                if node.is_healthy and node.state == SatelliteState.ADJUSTING:
                    node.state = SatelliteState.LISTENING

        if method in ("both", "ptp") and step % ptp_every == 0:
            master = int(sync.get("ptp_master_node_id", 0))
            adjacency = adjacency_from_edges(edges, ptp_nodes, sync)
            predecessor = shortest_paths_from_master(adjacency, master) if ptp_nodes[master].is_healthy else {}
            edge_type_lookup = {(min(edge.left, edge.right), max(edge.left, edge.right)): edge.edge_type for edge in edges}
            ptp_corrections: dict[int, float] = {}
            for slave_id, parent_id in predecessor.items():
                if rng.random() < float(sync.get("link_dropout_probability", 0.0)):
                    continue
                parent = ptp_nodes[parent_id]
                slave = ptp_nodes[slave_id]
                edge_type = edge_type_lookup.get((min(parent_id, slave_id), max(parent_id, slave_id)), "multi_hop")
                correction, record = exchange_ptp(parent, slave, edge_type, time_s, step, sync, rf_config, laser_config, rng)
                link_rows.append(record)
                rf_values.append(record["rf_ebno_db"])
                laser_values.append(record["laser_snr_db"])
                if correction is not None:
                    accepted_ptp += 1
                    ptp_corrections[slave_id] = float(sync.get("distributed_gain", 0.65)) * correction
            for node_id, correction in ptp_corrections.items():
                ptp_nodes[node_id].apply_correction(correction)
                if ptp_nodes[node_id].state == SatelliteState.ADJUSTING:
                    ptp_nodes[node_id].state = SatelliteState.LISTENING

        isdts_p2p, isdts_rms = compute_error_metrics(isdts_nodes)
        ptp_p2p, ptp_rms = compute_error_metrics(ptp_nodes)
        total_isdts += accepted_isdts
        total_ptp += accepted_ptp
        failed_count = sum(1 for node in isdts_nodes if not node.is_healthy)
        metrics_rows.append(
            {
                "step": step,
                "time_s": time_s,
                "isdts_peak_to_peak_error_ns": isdts_p2p if method in ("both", "isdts") else math.nan,
                "isdts_rms_error_ns": isdts_rms if method in ("both", "isdts") else math.nan,
                "ptp_peak_to_peak_error_ns": ptp_p2p if method in ("both", "ptp") else math.nan,
                "ptp_rms_error_ns": ptp_rms if method in ("both", "ptp") else math.nan,
                "mean_laser_snr_db": float(np.nanmean(laser_values)) if laser_values else math.nan,
                "mean_rf_ebno_db": float(np.nanmean(rf_values)) if rf_values else math.nan,
                "accepted_isdts_measurements": accepted_isdts,
                "accepted_ptp_measurements": accepted_ptp,
                "active_links": len(active_edges),
                "active_cross_plane_links": active_cross,
                "active_same_plane_links": active_same,
                "failed_nodes_count": failed_count,
            }
        )
        for method_name, nodes in (("isdts", isdts_nodes), ("ptp", ptp_nodes)):
            if method_name == "isdts" and method not in ("both", "isdts"):
                continue
            if method_name == "ptp" and method not in ("both", "ptp"):
                continue

            base_node_id = int(sync.get("ptp_master_node_id", 0))
            base_clock_offset_ns = nodes[base_node_id].clock_offset_ns

            for node in nodes:
                offset_rows.append(
                    {
                        "step": step,
                        "time_s": time_s,
                        "method": method_name,
                        "satellite_id": node.node_id,
                        "orbital_plane": node.orbital_plane,
                        "satellite_index": node.satellite_index,
                        "clock_offset_ns": node.clock_offset_ns,
                        "offset_from_base_clock_ns": node.clock_offset_ns - base_clock_offset_ns,
                        "state": node.state.value,
                        "latitude_deg": node.latitude_deg,
                    }
                )
        isdts_clock_history.append([node.clock_offset_ns for node in isdts_nodes])
        ptp_clock_history.append([node.clock_offset_ns for node in ptp_nodes])

    history = build_history(metrics_rows, isdts_clock_history, ptp_clock_history, config)
    write_parameters(config, run_dir / "parameters.json")
    write_csv(run_dir / sync.get("metrics_csv_name", "metrics.csv"), metrics_rows)
    write_csv(run_dir / sync.get("offsets_csv_name", "clock_offsets.csv"), offset_rows)
    write_csv(run_dir / sync.get("links_csv_name", "link_measurements.csv"), link_rows)
    write_csv(run_dir / "isdts_metrics.csv", [row for row in metrics_rows if not math.isnan(row["isdts_peak_to_peak_error_ns"])])
    write_csv(run_dir / "ptp_metrics.csv", [row for row in metrics_rows if not math.isnan(row["ptp_peak_to_peak_error_ns"])])

    threshold = float(sync.get("convergence_threshold_ns", 2.5))
    summary = make_summary(run_dir, parameter_path, metrics_rows, total_isdts, total_ptp, threshold, config)
    with (run_dir / sync.get("summary_json_name", "summary.json")).open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True, default=_json_default)
        file_obj.write("\n")

    generated_plots: list[Path] = []
    if not no_plots and bool(sync.get("save_plots", True)):
        generated_plots = create_plots(history, run_dir)
    return RunProducts(run_dir, history, summary, generated_plots, parameters_added)


def build_history(metrics_rows: list[dict[str, Any]], isdts_clock_history: list[list[float]], ptp_clock_history: list[list[float]], config: dict[str, Any]) -> dict[str, Any]:
    """Adapt metric rows to plots.py and fallback plotting formats."""

    steps = [row["step"] for row in metrics_rows]
    time = [row["time_s"] for row in metrics_rows]
    orbit_indices = [node_id // int(config["is_dts_simulation"]["satellites_per_plane"]) for node_id in range(int(config["is_dts_simulation"]["satellite_count"]))]
    return {
        "step": steps,
        "time_s": time,
        "time": time,
        "peak_to_peak_error_ns": [row["isdts_peak_to_peak_error_ns"] for row in metrics_rows],
        "rms_error_ns": [row["isdts_rms_error_ns"] for row in metrics_rows],
        "isdts_peak_to_peak_error_ns": [row["isdts_peak_to_peak_error_ns"] for row in metrics_rows],
        "isdts_rms_error_ns": [row["isdts_rms_error_ns"] for row in metrics_rows],
        "ptp_peak_to_peak_error_ns": [row["ptp_peak_to_peak_error_ns"] for row in metrics_rows],
        "ptp_rms_error_ns": [row["ptp_rms_error_ns"] for row in metrics_rows],
        "clock_offsets_ns": isdts_clock_history,
        "isdts_clock_offsets_ns": isdts_clock_history,
        "ptp_clock_offsets_ns": ptp_clock_history,
        "mean_laser_snr_db": [row["mean_laser_snr_db"] for row in metrics_rows],
        "mean_rf_ebno_db": [row["mean_rf_ebno_db"] for row in metrics_rows],
        "active_measurements": [row["accepted_isdts_measurements"] for row in metrics_rows],
        "accepted_ptp_measurements": [row["accepted_ptp_measurements"] for row in metrics_rows],
        "orbit_indices": orbit_indices,
    }


def convergence_time(rows: list[dict[str, Any]], key: str, threshold: float) -> float | None:
    """Return first time at or under convergence threshold."""

    for row in rows:
        value = row[key]
        if math.isfinite(value) and value <= threshold:
            return float(row["time_s"])
    return None


def make_summary(run_dir: Path, parameter_path: Path, rows: list[dict[str, Any]], total_isdts: int, total_ptp: int, threshold: float, config: dict[str, Any]) -> dict[str, Any]:
    """Create summary.json fields requested by the project."""

    final = rows[-1]
    return {
        "run_dir": str(run_dir),
        "parameter_file": str(parameter_path),
        "final_isdts_peak_to_peak_error_ns": final["isdts_peak_to_peak_error_ns"],
        "final_ptp_peak_to_peak_error_ns": final["ptp_peak_to_peak_error_ns"],
        "final_isdts_rms_error_ns": final["isdts_rms_error_ns"],
        "final_ptp_rms_error_ns": final["ptp_rms_error_ns"],
        "isdts_convergence_time_s": convergence_time(rows, "isdts_peak_to_peak_error_ns", threshold),
        "ptp_convergence_time_s": convergence_time(rows, "ptp_peak_to_peak_error_ns", threshold),
        "isdts_max_error_over_time_ns": float(np.nanmax([row["isdts_peak_to_peak_error_ns"] for row in rows])),
        "ptp_max_error_over_time_ns": float(np.nanmax([row["ptp_peak_to_peak_error_ns"] for row in rows])),
        "accepted_isdts_measurements_total": int(total_isdts),
        "accepted_ptp_measurements_total": int(total_ptp),
        "paper_expected_results": config["is_dts_simulation"].get("expected_results", {}),
        "notes": [
            "Circular ECI-like Walker propagation is deterministic and omits high-fidelity perturbations.",
            "RF and laser modules gate measurement acceptance and contribute timestamp-quality residuals; the IS-DTS correction uses the explicit paper timestamp equation.",
            "PTP baseline uses shortest-path hop-by-hop serial exchanges over the active topology.",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV, creating an empty file if there are no rows."""

    with path.open("w", newline="", encoding="utf-8") as file_obj:
        if not rows:
            file_obj.write("")
            return
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_plots(history: dict[str, Any], output_dir: Path) -> list[Path]:
    """Use plots.py where compatible, then create required fallback figures."""

    generated: set[Path] = set()
    try:
        plots = importlib.import_module("plots")
        if hasattr(plots, "create_run_plots"):
            generated.update(Path(path) for path in plots.create_run_plots(history, output_dir))
        if hasattr(plots, "plot_isdts_results"):
            plots.plot_isdts_results(
                history["time_s"],
                np.asarray(history["isdts_clock_offsets_ns"]) / 1e9,
                orbit_indices=history.get("orbit_indices"),
                save_path=output_dir / "isdts_results.png",
                final_errors=np.asarray(history["isdts_offset_from_base_clock_ns"])[-1] / 1e9,
            )
            generated.add(output_dir / "isdts_results.png")
        # plots.py exposes a paper Figure 8 helper with a richer signature; the
        # local fallback below always creates ptp_performance.png from this run.
    except Exception as exc:  # plotting is non-critical; fallbacks still run
        print(f"plots.py compatibility warning: {exc}")
    generated.update(create_fallback_plots(history, output_dir))
    return sorted(generated)


def create_fallback_plots(history: dict[str, Any], output_dir: Path) -> list[Path]:
    """Create required PNG files that are missing or incompatible in plots.py."""

    import matplotlib.pyplot as plt

    paths: list[Path] = []
    time = np.asarray(history["time_s"], dtype=float)

    def save(fig: Any, name: str) -> Path:
        path = output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    required = {
        "error_evolution.png",
        "clock_offsets.png",
        "link_quality.png",
        "isdts_results.png",
        "ptp_performance.png",
        "isdts_vs_ptp_comparison.png",
    }

    if not (output_dir / "isdts_vs_ptp_comparison.png").exists():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(time, history["isdts_peak_to_peak_error_ns"], label="IS-DTS peak-to-peak")
        ax.plot(time, history["ptp_peak_to_peak_error_ns"], label="PTP peak-to-peak")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Peak-to-peak clock error [ns]")
        ax.set_title("IS-DTS vs traditional PTP")
        ax.grid(True, alpha=0.3)
        ax.legend()
        paths.append(save(fig, "isdts_vs_ptp_comparison.png"))

    for name in required:
        path = output_dir / name
        if path.exists():
            paths.append(path)
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        if name == "clock_offsets.png":
            offsets = np.asarray(history["isdts_clock_offsets_ns"], dtype=float)
            for idx in range(offsets.shape[1]):
                ax.plot(time, offsets[:, idx], linewidth=0.7, alpha=0.6)
            ax.set_ylabel("Clock offset [ns]")
            ax.set_title("IS-DTS satellite clock offsets")
        elif name == "link_quality.png":
            ax.plot(time, history["mean_laser_snr_db"], label="Laser SNR [dB]")
            ax.plot(time, history["mean_rf_ebno_db"], label="RF Eb/No [dB]")
            ax.legend()
            ax.set_ylabel("Link quality [dB]")
            ax.set_title("RF and laser link quality")
        elif name == "ptp_performance.png":
            ax.plot(time, history["ptp_peak_to_peak_error_ns"], label="PTP peak-to-peak")
            ax.plot(time, history["ptp_rms_error_ns"], label="PTP RMS")
            ax.legend()
            ax.set_ylabel("Clock error [ns]")
            ax.set_title("Traditional PTP baseline")
        else:
            ax.plot(time, history["isdts_peak_to_peak_error_ns"], label="IS-DTS peak-to-peak")
            ax.plot(time, history["isdts_rms_error_ns"], label="IS-DTS RMS")
            ax.legend()
            ax.set_ylabel("Clock error [ns]")
            ax.set_title("IS-DTS convergence")
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)
        paths.append(save(fig, name))
    return paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run the IS-DTS/RF/laser satellite synchronization simulation.")
    parser.add_argument("--parameters", default=None, help="Path to parameters, parameters.json, or parameters.py")
    parser.add_argument("--method", choices=("both", "isdts", "ptp"), default="both", help="Synchronization method(s) to run")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config, parameter_path = load_parameters(args.parameters)
    parameters_added = ensure_parameter_defaults(config, parameter_path)
    # Fill computed values for this run snapshot without overwriting explicit null in source unless it was missing.
    sync = config["is_dts_simulation"]
    if sync.get("orbital_period_s") is None:
        sync["computed_orbital_period_s"] = orbital_period_s(sync)
    validate_parameters(config)
    run_sanity_checks(config)
    products = run_simulation(config, parameter_path, args.method, args.no_plots, parameters_added)
    summary = products.summary
    print(f"Output directory: {products.run_dir}")
    print(f"Final IS-DTS peak-to-peak error: {summary['final_isdts_peak_to_peak_error_ns']:.6g} ns")
    print(f"Final PTP peak-to-peak error: {summary['final_ptp_peak_to_peak_error_ns']:.6g} ns")
    print(f"IS-DTS convergence time: {summary['isdts_convergence_time_s']}")
    print(f"PTP convergence time: {summary['ptp_convergence_time_s']}")
    print(f"Generated plots: {len(products.generated_plots)}")


if __name__ == "__main__":
    main()
