"""Simplified Precision Time Protocol baseline."""

from __future__ import annotations

from dataclasses import dataclass

from src.network import SatelliteNetwork
from src.satellite import Satellite


@dataclass
class PTPConfig:
    """Tuning parameters for a master-slave baseline."""

    master_id: int = 0
    correction_gain: float = 0.75
    measurement_noise_ns: float = 1.0


class SimplifiedPTP:
    """Serial master-slave synchronization baseline.

    Each healthy non-master node exchanges a two-way timestamp with the master.
    The classic symmetric-delay assumption leaves a bias of half the path-delay
    asymmetry, which is deliberately retained to illustrate PTP degradation when
    satellite link delays vary by direction during sequential exchanges.
    """

    def __init__(self, config: PTPConfig | None = None) -> None:
        self.config = config or PTPConfig()

    def step(
        self,
        satellites: list[Satellite],
        network: SatelliteNetwork,
        true_time_s: float,
    ) -> None:
        """Execute one serial master-to-slave synchronization cycle."""

        master = satellites[self.config.master_id]
        if not master.is_healthy:
            return

        for node_id, satellite in enumerate(satellites):
            if node_id == self.config.master_id or not satellite.is_healthy:
                continue
            sample = self._sample_path_delay(network, self.config.master_id, node_id)
            if sample is None:
                continue
            delay_master_to_node, delay_node_to_master = sample
            asymmetry_bias = (delay_master_to_node - delay_node_to_master) / 2.0
            noise = float(network.rng.normal(0.0, self.config.measurement_noise_ns))
            estimated_master_minus_node = (
                master.offset_ns - satellite.offset_ns + asymmetry_bias + noise
            )
            satellite.apply_correction(
                self.config.correction_gain * estimated_master_minus_node
            )

    def _sample_path_delay(
        self, network: SatelliteNetwork, source: int, target: int
    ) -> tuple[float, float] | None:
        """Sample directed delay along the shortest currently active path."""

        path = self._shortest_path(network, source, target)
        if path is None:
            return None

        forward_delay = 0.0
        reverse_delay = 0.0
        for left, right in zip(path, path[1:]):
            link = network.sample_link(left, right)
            forward_delay += link.delay_ij_ns
            reverse_delay += link.delay_ji_ns
        return forward_delay, reverse_delay

    @staticmethod
    def _shortest_path(
        network: SatelliteNetwork, source: int, target: int
    ) -> list[int] | None:
        """Breadth-first path search over active links."""

        queue: list[list[int]] = [[source]]
        visited = {source}
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for neighbor in sorted(network.neighbors(node)):
                if neighbor in visited:
                    continue
                next_path = [*path, neighbor]
                if neighbor == target:
                    return next_path
                visited.add(neighbor)
                queue.append(next_path)
        return None
