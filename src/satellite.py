"""Satellite node model used by the synchronization simulations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SatelliteState(str, Enum):
    """Operational state of a simulated satellite clock node."""

    INITIALIZING = "Initializing"
    LISTENING = "Listening"
    ADJUSTING = "Adjusting"
    FAULTY = "Faulty"


@dataclass
class Satellite:
    """A satellite with a simple affine local clock.

    The simulation keeps the true/reference time separate from each local clock.
    ``offset_ns`` is the current clock phase error relative to true time, while
    ``drift_ns_per_s`` is the oscillator frequency error. A positive offset means
    the satellite clock is ahead of true time.
    """

    node_id: int
    offset_ns: float
    drift_ns_per_s: float
    state: SatelliteState = SatelliteState.INITIALIZING
    correction_gain: float = 1.0

    def local_time_ns(self, true_time_s: float) -> float:
        """Return the satellite's local timestamp in nanoseconds."""

        return true_time_s * 1e9 + self.offset_ns

    def tick(self, dt_s: float) -> None:
        """Advance the clock error according to oscillator drift."""

        if self.state != SatelliteState.FAULTY:
            self.offset_ns += self.drift_ns_per_s * dt_s

    def complete_initialization(self) -> None:
        """Move a healthy node into the message-listening state."""

        if self.state == SatelliteState.INITIALIZING:
            self.state = SatelliteState.LISTENING

    def apply_correction(self, correction_ns: float) -> None:
        """Adjust the local clock phase by a bounded correction."""

        if self.state == SatelliteState.FAULTY:
            return
        self.state = SatelliteState.ADJUSTING
        self.offset_ns += self.correction_gain * correction_ns
        self.state = SatelliteState.LISTENING

    def mark_faulty(self) -> None:
        """Remove this node from future synchronization calculations."""

        self.state = SatelliteState.FAULTY

    @property
    def is_healthy(self) -> bool:
        """Whether the node should participate in synchronization."""

        return self.state != SatelliteState.FAULTY
