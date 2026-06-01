"""RF satellite receiver link-budget and synchronization simulation.

The module is inspired by MathWorks' RF satellite link example and adapts the
same high-level receive chain to this repository's educational synchronization
models: a satellite transmits time-tagged frames, the RF channel applies free
space path loss, Doppler, thermal noise, oscillator phase noise, and receiver
noise figure, and the receiver estimates link quality plus timestamp error.

The implementation intentionally stays dependency-light (NumPy only) so it can
be used in scripts and tests without Communications Toolbox equivalents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.satellite import Satellite, SatelliteState

BOLTZMANN_W_PER_HZ_K = 1.380649e-23
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


def db_to_linear(value_db: float) -> float:
    """Convert a decibel value to linear scale."""

    return 10.0 ** (value_db / 10.0)


def linear_to_db(value: float, floor: float = 1e-300) -> float:
    """Convert a linear value to decibels with a numerical floor."""

    return 10.0 * np.log10(max(float(value), floor))


def free_space_path_loss_db(range_m: float, carrier_frequency_hz: float) -> float:
    """Return free-space path loss for a one-way RF satellite link."""

    if range_m <= 0.0:
        raise ValueError("range_m must be positive")
    if carrier_frequency_hz <= 0.0:
        raise ValueError("carrier_frequency_hz must be positive")
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / carrier_frequency_hz
    return linear_to_db((4.0 * np.pi * range_m / wavelength_m) ** 2)


def doppler_shift_hz(relative_velocity_m_per_s: float, carrier_frequency_hz: float) -> float:
    """Compute the first-order Doppler frequency shift for radial velocity."""

    return carrier_frequency_hz * relative_velocity_m_per_s / SPEED_OF_LIGHT_M_PER_S


@dataclass(frozen=True)
class RFReceiverConfig:
    """Configuration for the RF satellite receiver model."""

    carrier_frequency_hz: float = 2.2e9
    sample_rate_hz: float = 1.0e6
    symbol_rate_hz: float = 250.0e3
    transmit_power_dbw: float = 8.0
    transmitter_gain_dbi: float = 24.0
    receiver_gain_dbi: float = 32.0
    system_loss_db: float = 2.0
    receiver_noise_figure_db: float = 3.0
    antenna_temperature_k: float = 290.0
    implementation_loss_db: float = 1.0
    sync_threshold_db: float = 6.0
    timestamp_jitter_std_s: float = 2.5e-9
    phase_noise_std_rad: float = 0.01
    frame_symbols: int = 512

    @property
    def samples_per_symbol(self) -> int:
        """Integer oversampling ratio used by the educational waveform model."""

        ratio = self.sample_rate_hz / self.symbol_rate_hz
        rounded = int(round(ratio))
        if rounded <= 0 or not np.isclose(ratio, rounded):
            raise ValueError("sample_rate_hz must be an integer multiple of symbol_rate_hz")
        return rounded

    @property
    def noise_bandwidth_hz(self) -> float:
        """Approximate receiver noise bandwidth."""

        return self.symbol_rate_hz


@dataclass(frozen=True)
class RFLinkGeometry:
    """One RF link snapshot between a transmitter and receiver."""

    range_m: float = 1_200_000.0
    relative_velocity_m_per_s: float = 0.0


@dataclass(frozen=True)
class RFReceiverMeasurement:
    """Outputs produced by one received satellite frame."""

    sender_id: int
    receiver_id: int
    transmit_time_ns: float
    receive_time_ns: float
    propagation_delay_ns: float
    doppler_hz: float
    path_loss_db: float
    received_power_dbw: float
    cnr_db: float
    ebno_db: float
    estimated_offset_ns: float
    synchronized: bool
    bit_errors: int
    ber: float


@dataclass
class SatelliteRFReceiver:
    """Receiver that tracks satellite time messages over an RF channel.

    This class follows the structure of a MATLAB-style satellite receiver demo:
    build a frame, apply an RF satellite channel, estimate carrier/link metrics,
    demodulate, and turn the recovered frame timestamp into a clock-offset
    correction. It reuses the repository's :class:`src.satellite.Satellite`
    state machine, moving healthy initialized nodes into Listening/Adjusting
    states as measurements are processed.
    """

    receiver: Satellite
    config: RFReceiverConfig = field(default_factory=RFReceiverConfig)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    measurements: list[RFReceiverMeasurement] = field(default_factory=list)

    def receive_time_frame(
        self,
        sender: Satellite,
        true_time_s: float,
        geometry: RFLinkGeometry | None = None,
    ) -> RFReceiverMeasurement:
        """Receive one time-tagged RF frame from ``sender``.

        Parameters are deliberately compact so the receiver can be dropped into
        the existing synchronization examples. ``true_time_s`` is the reference
        time at which the sender emits its frame.
        """

        if geometry is None:
            geometry = RFLinkGeometry()
        if sender.state == SatelliteState.INITIALIZING:
            sender.complete_initialization()
        if self.receiver.state == SatelliteState.INITIALIZING:
            self.receiver.complete_initialization()

        transmit_time_ns = sender.local_time_ns(true_time_s)
        propagation_delay_s = geometry.range_m / SPEED_OF_LIGHT_M_PER_S
        ideal_receive_true_time_s = true_time_s + propagation_delay_s
        receive_time_ns = self.receiver.local_time_ns(ideal_receive_true_time_s)

        path_loss_db = free_space_path_loss_db(
            geometry.range_m, self.config.carrier_frequency_hz
        )
        received_power_dbw = self._received_power_dbw(path_loss_db)
        cnr_db = self._cnr_db(received_power_dbw)
        ebno_db = cnr_db - linear_to_db(self.config.symbol_rate_hz / self.config.noise_bandwidth_hz)
        doppler_hz = doppler_shift_hz(
            geometry.relative_velocity_m_per_s, self.config.carrier_frequency_hz
        )

        bits = self._build_frame_bits(sender.node_id, transmit_time_ns)
        demodulated_bits = self._transmit_through_baseband_channel(bits, ebno_db, doppler_hz)
        bit_errors = int(np.count_nonzero(bits != demodulated_bits))
        ber = bit_errors / bits.size

        timestamp_noise_ns = self._timestamp_noise_ns(ebno_db)
        estimated_offset_ns = (
            receive_time_ns - transmit_time_ns - propagation_delay_s * 1e9 + timestamp_noise_ns
        )
        synchronized = bool(ebno_db >= self.config.sync_threshold_db and bit_errors == 0)

        measurement = RFReceiverMeasurement(
            sender_id=sender.node_id,
            receiver_id=self.receiver.node_id,
            transmit_time_ns=transmit_time_ns,
            receive_time_ns=receive_time_ns,
            propagation_delay_ns=propagation_delay_s * 1e9,
            doppler_hz=doppler_hz,
            path_loss_db=path_loss_db,
            received_power_dbw=received_power_dbw,
            cnr_db=cnr_db,
            ebno_db=ebno_db,
            estimated_offset_ns=estimated_offset_ns,
            synchronized=synchronized,
            bit_errors=bit_errors,
            ber=ber,
        )
        self.measurements.append(measurement)
        return measurement

    def apply_last_correction(self, gain: float = 0.5) -> None:
        """Correct the receiver clock using the latest synchronized measurement."""

        if not self.measurements:
            return
        measurement = self.measurements[-1]
        if measurement.synchronized:
            self.receiver.apply_correction(-gain * measurement.estimated_offset_ns)

    def _received_power_dbw(self, path_loss_db: float) -> float:
        return (
            self.config.transmit_power_dbw
            + self.config.transmitter_gain_dbi
            + self.config.receiver_gain_dbi
            - path_loss_db
            - self.config.system_loss_db
            - self.config.implementation_loss_db
        )

    def _cnr_db(self, received_power_dbw: float) -> float:
        noise_temperature_k = self.config.antenna_temperature_k * db_to_linear(
            self.config.receiver_noise_figure_db
        )
        noise_power_w = BOLTZMANN_W_PER_HZ_K * noise_temperature_k * self.config.noise_bandwidth_hz
        return received_power_dbw - linear_to_db(noise_power_w)

    def _timestamp_noise_ns(self, ebno_db: float) -> float:
        # Better Eb/No produces tighter timestamp estimates, while oscillator
        # jitter remains as a floor. The scale is intentionally simple and stable.
        ebno_linear = db_to_linear(ebno_db)
        tracking_std_ns = 1e9 / (2.0 * np.pi * self.config.symbol_rate_hz * np.sqrt(max(ebno_linear, 1e-12)))
        jitter_std_ns = self.config.timestamp_jitter_std_s * 1e9
        return float(self.rng.normal(0.0, np.hypot(tracking_std_ns, jitter_std_ns)))

    def _build_frame_bits(self, sender_id: int, transmit_time_ns: float) -> np.ndarray:
        preamble = np.array([1, 0, 1, 0, 1, 1, 0, 0] * 4, dtype=np.uint8)
        payload = np.zeros(self.config.frame_symbols - preamble.size, dtype=np.uint8)
        seed = (int(sender_id) * 1_000_003 + int(round(transmit_time_ns))) % (2**32)
        payload_rng = np.random.default_rng(seed)
        payload[:] = payload_rng.integers(0, 2, size=payload.size, dtype=np.uint8)
        return np.concatenate([preamble, payload])

    def _transmit_through_baseband_channel(
        self, bits: np.ndarray, ebno_db: float, doppler_hz: float
    ) -> np.ndarray:
        # BPSK symbols with phase noise and a deterministic carrier rotation from
        # Doppler. Hard decisions approximate a simple satellite receiver chain.
        symbols = 2.0 * bits.astype(float) - 1.0
        symbol_times = np.arange(bits.size, dtype=float) / self.config.symbol_rate_hz
        phase = (
            2.0 * np.pi * doppler_hz * symbol_times
            + self.rng.normal(0.0, self.config.phase_noise_std_rad, bits.size)
        )
        # A practical receiver estimates and removes the carrier frequency/phase.
        # The residual phase error below keeps Doppler visible without making the
        # educational hard-decision detector fail solely from uncorrected rotation.
        corrected_phase = phase - 2.0 * np.pi * doppler_hz * symbol_times
        rotated = symbols * np.cos(corrected_phase)
        noise_std = np.sqrt(1.0 / (2.0 * db_to_linear(ebno_db)))
        noisy = rotated + self.rng.normal(0.0, noise_std, bits.size)
        return (noisy >= 0.0).astype(np.uint8)


def simulate_satellite_receiver(
    steps: int = 5,
    seed: int = 11,
    range_m: float = 1_200_000.0,
    relative_velocity_m_per_s: float = 250.0,
) -> list[RFReceiverMeasurement]:
    """Run a compact RF receiver demo using two repository satellites."""

    rng = np.random.default_rng(seed)
    transmitter = Satellite(node_id=0, offset_ns=250.0, drift_ns_per_s=0.2)
    receiver = Satellite(node_id=1, offset_ns=-750.0, drift_ns_per_s=-0.1)
    rf_receiver = SatelliteRFReceiver(receiver=receiver, rng=rng)
    geometry = RFLinkGeometry(range_m=range_m, relative_velocity_m_per_s=relative_velocity_m_per_s)

    results: list[RFReceiverMeasurement] = []
    for step in range(steps):
        true_time_s = float(step)
        transmitter.tick(1.0)
        receiver.tick(1.0)
        measurement = rf_receiver.receive_time_frame(transmitter, true_time_s, geometry)
        rf_receiver.apply_last_correction(gain=0.5)
        results.append(measurement)
    return results


if __name__ == "__main__":
    for item in simulate_satellite_receiver():
        print(
            f"tx={item.sender_id} rx={item.receiver_id} "
            f"Eb/No={item.ebno_db:.2f} dB BER={item.ber:.3g} "
            f"offset={item.estimated_offset_ns:.2f} ns sync={item.synchronized}"
        )
