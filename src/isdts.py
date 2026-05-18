"""Distributed IS-DTS-style synchronization algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.network import SatelliteNetwork
from src.satellite import Satellite


@dataclass
class ISDTSConfig:
    """Tuning parameters for the distributed synchronization update."""

    correction_gain: float = 0.55
    measurement_noise_ns: float = 0.5
    faulty_weight: float = 0.0


class ISDTSAlgorithm:
    """Parallel neighbor-averaging synchronization.

    At each simulation step every healthy satellite simultaneously exchanges
    timestamps with active neighbors. The pairwise estimate is modeled as the
    neighbor's clock error minus the local clock error plus residual measurement
    noise. Each node then applies a consensus-like correction equal to the average
    of its pairwise offset estimates.
    """

    def __init__(self, config: ISDTSConfig | None = None) -> None:
        self.config = config or ISDTSConfig()

    def step(
        self,
        satellites: list[Satellite],
        network: SatelliteNetwork,
        true_time_s: float,
    ) -> None:
        """Execute one parallel distributed synchronization round."""

        corrections = np.zeros(len(satellites), dtype=float)
        weights = np.zeros(len(satellites), dtype=float)

        for i, j in network.active_edges():
            sat_i = satellites[i]
            sat_j = satellites[j]
            if not sat_i.is_healthy and not sat_j.is_healthy:
                continue

            sample = network.sample_link(i, j)
            # Parallel two-way exchange suppresses most common propagation delay;
            # residual error is driven by delay asymmetry and timestamp noise.
            asymmetry_error = (sample.delay_ij_ns - sample.delay_ji_ns) / 2.0
            noise = float(network.rng.normal(0.0, self.config.measurement_noise_ns))

            if sat_i.is_healthy and sat_j.is_healthy:
                offset_j_minus_i = sat_j.offset_ns - sat_i.offset_ns + asymmetry_error + noise
                corrections[i] += offset_j_minus_i
                corrections[j] -= offset_j_minus_i
                weights[i] += 1.0
                weights[j] += 1.0
            elif sat_i.is_healthy:
                weights[i] += self.config.faulty_weight
            elif sat_j.is_healthy:
                weights[j] += self.config.faulty_weight

        for idx, satellite in enumerate(satellites):
            if satellite.is_healthy and weights[idx] > 0.0:
                average_offset = corrections[idx] / weights[idx]
                satellite.apply_correction(self.config.correction_gain * average_offset)
