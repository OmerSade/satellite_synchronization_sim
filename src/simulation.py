"""Main simulation loop for comparing IS-DTS and a PTP-like baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.isdts import ISDTSAlgorithm, ISDTSConfig
from src.network import SatelliteNetwork
from src.ptp import PTPConfig, SimplifiedPTP
from src.satellite import Satellite
from utils.metrics import max_time_difference_ns, rms_error_ns, time_to_convergence


@dataclass
class SimulationConfig:
    """Experiment configuration."""

    num_satellites: int = 72
    steps: int = 120
    dt_s: float = 1.0
    initial_offset_std_ns: float = 1_000.0
    drift_std_ns_per_s: float = 2.0
    seed: int = 7
    convergence_threshold_ns: float = 10.0
    link_toggle_probability: float = 0.0
    failure_events: dict[int, list[int]] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Time series and summary metrics for one comparison run."""

    times_s: list[float]
    isdts_max_error_ns: list[float]
    ptp_max_error_ns: list[float]
    isdts_rms_error_ns: list[float]
    ptp_rms_error_ns: list[float]
    isdts_convergence_s: float | None
    ptp_convergence_s: float | None


class SynchronizationSimulation:
    """Create two identical constellations and compare synchronization methods."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.isdts_network = SatelliteNetwork.leo_mesh(
            self.config.num_satellites, self.rng
        )
        # A separate RNG keeps the PTP run reproducible but independent.
        ptp_rng = np.random.default_rng(self.config.seed + 1)
        self.ptp_network = SatelliteNetwork(
            num_nodes=self.isdts_network.num_nodes,
            adjacency={node: set(nbrs) for node, nbrs in self.isdts_network.adjacency.items()},
            rng=ptp_rng,
            asymmetry_std_ns=self.isdts_network.asymmetry_std_ns,
            jitter_std_ns=self.isdts_network.jitter_std_ns,
            edge_base_delay=self.isdts_network.base_delays(),
        )
        self.isdts_satellites, self.ptp_satellites = self._create_satellite_copies()
        self.isdts = ISDTSAlgorithm(ISDTSConfig())
        self.ptp = SimplifiedPTP(PTPConfig())

    def _create_satellite_copies(self) -> tuple[list[Satellite], list[Satellite]]:
        offsets = self.rng.normal(
            0.0, self.config.initial_offset_std_ns, self.config.num_satellites
        )
        drifts = self.rng.normal(
            0.0, self.config.drift_std_ns_per_s, self.config.num_satellites
        )
        isdts_satellites = [
            Satellite(node_id=i, offset_ns=float(offsets[i]), drift_ns_per_s=float(drifts[i]))
            for i in range(self.config.num_satellites)
        ]
        ptp_satellites = [
            Satellite(node_id=i, offset_ns=float(offsets[i]), drift_ns_per_s=float(drifts[i]))
            for i in range(self.config.num_satellites)
        ]
        for satellite in [*isdts_satellites, *ptp_satellites]:
            satellite.complete_initialization()
        return isdts_satellites, ptp_satellites

    def run(self) -> SimulationResult:
        """Run the full discrete-time simulation."""

        times_s: list[float] = []
        isdts_max: list[float] = []
        ptp_max: list[float] = []
        isdts_rms: list[float] = []
        ptp_rms: list[float] = []

        for step in range(self.config.steps):
            true_time_s = step * self.config.dt_s
            self._apply_fault_events(step)
            self._apply_topology_changes()

            for satellite in self.isdts_satellites:
                satellite.tick(self.config.dt_s)
            for satellite in self.ptp_satellites:
                satellite.tick(self.config.dt_s)

            self.isdts.step(self.isdts_satellites, self.isdts_network, true_time_s)
            self.ptp.step(self.ptp_satellites, self.ptp_network, true_time_s)

            times_s.append(true_time_s)
            isdts_max.append(max_time_difference_ns(self.isdts_satellites))
            ptp_max.append(max_time_difference_ns(self.ptp_satellites))
            isdts_rms.append(rms_error_ns(self.isdts_satellites))
            ptp_rms.append(rms_error_ns(self.ptp_satellites))

        return SimulationResult(
            times_s=times_s,
            isdts_max_error_ns=isdts_max,
            ptp_max_error_ns=ptp_max,
            isdts_rms_error_ns=isdts_rms,
            ptp_rms_error_ns=ptp_rms,
            isdts_convergence_s=time_to_convergence(
                times_s, isdts_max, self.config.convergence_threshold_ns
            ),
            ptp_convergence_s=time_to_convergence(
                times_s, ptp_max, self.config.convergence_threshold_ns
            ),
        )

    def _apply_fault_events(self, step: int) -> None:
        for node_id in self.config.failure_events.get(step, []):
            self.isdts_satellites[node_id].mark_faulty()
            self.ptp_satellites[node_id].mark_faulty()

    def _apply_topology_changes(self) -> None:
        probability = self.config.link_toggle_probability
        if probability <= 0.0:
            return
        self.isdts_network.randomly_toggle_links(probability)
        self.ptp_network.randomly_toggle_links(probability)
