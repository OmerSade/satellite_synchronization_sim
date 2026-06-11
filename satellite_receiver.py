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

import argparse
import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class SatelliteState(Enum):
    """Minimal state machine compatible with IS_DTS.py satellite nodes."""

    INITIALIZING = "Initializing"
    LISTENING = "Listening"
    ADJUSTING = "Adjusting"
    FAULTY = "Faulty"


@dataclass
class Satellite:
    """Small local satellite clock used only for direct script execution."""

    node_id: int
    offset_ns: float = 0.0
    drift_ns_per_s: float = 0.0
    state: SatelliteState = SatelliteState.INITIALIZING

    @property
    def is_healthy(self) -> bool:
        return self.state != SatelliteState.FAULTY

    def complete_initialization(self) -> None:
        if self.state == SatelliteState.INITIALIZING:
            self.state = SatelliteState.LISTENING

    def tick(self, dt_s: float) -> None:
        if self.is_healthy:
            self.offset_ns += self.drift_ns_per_s * dt_s

    def local_time_ns(self, true_time_s: float) -> float:
        return true_time_s * 1e9 + self.offset_ns

    def apply_correction(self, correction_ns: float) -> None:
        if self.is_healthy:
            self.state = SatelliteState.ADJUSTING
            self.offset_ns += float(correction_ns)


BOLTZMANN_W_PER_HZ_K = 1.380649e-23
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0

def create_demo_satellite(
    node_id: int, offset_ns: float, drift_ns_per_s: float
) -> Satellite:
    """Create a demo satellite across both standalone and IS_DTS compatibility types."""

    try:
        return Satellite(
            node_id=node_id, offset_ns=offset_ns, drift_ns_per_s=drift_ns_per_s
        )
    except TypeError:
        # IS_DTS.py exposes SatelliteNode through src.satellite; its constructor
        # carries orbit metadata and a clock_offset_ns name, while retaining the
        # same runtime methods used by these demos.
        return Satellite(
            node_id=node_id,
            orbital_plane=0,
            satellite_index=node_id,
            clock_offset_ns=offset_ns,
            drift_ns_per_s=drift_ns_per_s,
        )

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


def doppler_shift_hz(
    relative_velocity_m_per_s: float, carrier_frequency_hz: float
) -> float:
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
        doppler_hz = doppler_shift_hz(geometry.relative_velocity_m_per_s, self.config.carrier_frequency_hz)

        bits = self._build_frame_bits(sender.node_id, transmit_time_ns)
        demodulated_bits = self._transmit_through_baseband_channel(bits, ebno_db, doppler_hz)
        bit_errors = int(np.count_nonzero(bits != demodulated_bits))
        ber = bit_errors / bits.size

        timestamp_noise_ns = self._timestamp_noise_ns(ebno_db)
        estimated_offset_ns = (receive_time_ns - transmit_time_ns - propagation_delay_s * 1e9 + timestamp_noise_ns)
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
        return (self.config.transmit_power_dbw + self.config.transmitter_gain_dbi + self.config.receiver_gain_dbi - path_loss_db - self.config.system_loss_db - self.config.implementation_loss_db)

    def _cnr_db(self, received_power_dbw: float) -> float:
        noise_temperature_k = self.config.antenna_temperature_k * db_to_linear(self.config.receiver_noise_figure_db)
        noise_power_w = (BOLTZMANN_W_PER_HZ_K * noise_temperature_k * self.config.noise_bandwidth_hz)
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
        phase = 2.0 * np.pi * doppler_hz * symbol_times + self.rng.normal(0.0, self.config.phase_noise_std_rad, bits.size)
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
    transmitter = create_demo_satellite(node_id=0, offset_ns=250.0, drift_ns_per_s=0.2)
    receiver = create_demo_satellite(node_id=1, offset_ns=-750.0, drift_ns_per_s=-0.1)
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


@dataclass(frozen=True)
class RFSatelliteLinkDemoConfig:
    """Configuration for the standalone MATLAB-style RF satellite link demo."""

    steps: int = 20
    seed: int = 11
    output_dir: Path = Path("outputs/rf_satellite_link_demo")
    altitude_km: float = 35_600.0
    frequency_mhz: float = 4_000.0
    tx_dish_diameter_m: float = 0.4
    rx_dish_diameter_m: float = 0.4
    noise_temperature_k: float = 20.0
    hpa_backoff_db: float = 30.0
    doppler_error_hz: float = 0.0
    phase_noise_level: str = "negligible"
    iq_impairment: str = "none"
    digital_predistortion: bool = False
    dc_offset_correction: bool = False
    doppler_correction: bool = False
    iq_correction: bool = False
    make_plots: bool = True

@dataclass(frozen=True)
class RFSatelliteLinkDemoConfig:
    """Configuration for the standalone MATLAB-style RF satellite link demo."""

    steps: int = 20
    seed: int = 11
    output_dir: Path = Path("outputs/rf_satellite_link_demo")
    altitude_km: float = 35_600.0
    frequency_mhz: float = 4_000.0
    tx_dish_diameter_m: float = 0.4
    rx_dish_diameter_m: float = 0.4
    noise_temperature_k: float = 20.0
    hpa_backoff_db: float = 30.0
    doppler_error_hz: float = 0.0
    phase_noise_level: str = "negligible"
    iq_impairment: str = "none"
    digital_predistortion: bool = False
    dc_offset_correction: bool = False
    doppler_correction: bool = False
    iq_correction: bool = False
    make_plots: bool = True


@dataclass(frozen=True)
class RFBasebandStepResult:
    """Numerically consistent symbol-rate RF baseband result for one demo step."""

    bits: np.ndarray
    symbols: np.ndarray
    hpa_symbols: np.ndarray
    received_symbols: np.ndarray
    received_symbols_agc: np.ndarray
    noise_symbols: np.ndarray
    bit_errors: int
    ber: float
    evm_percent: float
    evm_snr_db: float
    measured_baseband_snr_db: float
    agc_alpha: complex
    hpa_probe_in: np.ndarray
    hpa_probe_am: np.ndarray
    hpa_probe_pm_deg: np.ndarray


_PHASE_NOISE_STD_RAD = {"negligible": 0.0, "low": 0.04, "high": 0.18}
_SUPPORTED_IQ_IMPAIRMENTS = (
    "none",
    "amplitude_imbalance_3db",
    "phase_imbalance_20deg",
    "i_dc_offset_1e-8",
    "q_dc_offset_5e-8",
)
_BITS_PER_QAM16_SYMBOL = 4
_QAM16_NORMALIZATION = np.sqrt(10.0)
_QAM16_LEVELS = np.array([-3.0, -1.0, 1.0, 3.0]) / _QAM16_NORMALIZATION
_QAM16_LEVEL_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)


def parabolic_dish_gain_dbi(
    diameter_m: float, frequency_hz: float, efficiency: float = 0.60
) -> float:
    """Return idealized parabolic dish gain in dBi."""

    wavelength_m = SPEED_OF_LIGHT_M_PER_S / frequency_hz
    return linear_to_db(efficiency * (np.pi * diameter_m / wavelength_m) ** 2)


def qam16_modulate(bits: np.ndarray) -> np.ndarray:
    """Map bits to unit-average-power Gray-coded 16-QAM symbols."""

    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % _BITS_PER_QAM16_SYMBOL:
        raise ValueError("16-QAM modulation requires a multiple of 4 bits")
    groups = bits.reshape(-1, _BITS_PER_QAM16_SYMBOL)
    # Gray-coded two-bit amplitudes: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3.
    i_index = np.where(groups[:, 0] == 0, groups[:, 1], 3 - groups[:, 1])
    q_index = np.where(groups[:, 2] == 0, groups[:, 3], 3 - groups[:, 3])
    return _QAM16_LEVELS[i_index] + 1j * _QAM16_LEVELS[q_index]


def qam16_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Nearest-neighbor demodulator matching :func:`qam16_modulate`."""

    symbols = np.asarray(symbols, dtype=complex)
    i_index = np.argmin(
        np.abs(np.real(symbols)[:, None] - _QAM16_LEVELS[None, :]), axis=1
    )
    q_index = np.argmin(
        np.abs(np.imag(symbols)[:, None] - _QAM16_LEVELS[None, :]), axis=1
    )
    decoded = np.empty((symbols.size, _BITS_PER_QAM16_SYMBOL), dtype=np.uint8)
    decoded[:, :2] = _QAM16_LEVEL_BITS[i_index]
    decoded[:, 2:] = _QAM16_LEVEL_BITS[q_index]
    return decoded.reshape(-1)


def educational_pulse_shape(
    symbols: np.ndarray, samples_per_symbol: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a short normalized pulse-shaping filter for spectrum display only.

    The default BER/EVM path intentionally runs at symbol rate to avoid hiding
    timing/group-delay mistakes inside an educational approximation. This helper
    is still useful for MATLAB-like spectrum plots.
    """

    span = 8
    t = (
        np.arange(-span * samples_per_symbol, span * samples_per_symbol + 1)
        / samples_per_symbol
    )
    beta = 0.35
    pulse = np.sinc(t) * np.hamming(t.size) * np.cos(np.pi * beta * t)
    pulse /= np.sqrt(np.sum(pulse**2))
    upsampled = np.zeros(symbols.size * samples_per_symbol, dtype=complex)
    upsampled[::samples_per_symbol] = symbols
    return np.convolve(upsampled, pulse, mode="same"), pulse


def matched_filter_and_downsample(
    waveform: np.ndarray, pulse: np.ndarray, samples_per_symbol: int = 4
) -> np.ndarray:
    """Matched-filter a shaped waveform and sample at symbol centers."""

    filtered = np.convolve(waveform, pulse, mode="same")
    return filtered[::samples_per_symbol]


def apply_hpa_nonlinearity(
    signal: np.ndarray, backoff_db: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a memoryless HPA whose input backoff controls saturation distance.

    ``backoff_db`` is treated as input backoff relative to the average unit-power
    constellation. At 30 dB, the saturation amplitude is far from the signal and
    distortion is negligible; at 7 dB it is visible; at 1 dB it is severe.
    """

    input_backoff_linear = db_to_linear(backoff_db)
    saturation_amplitude = np.sqrt(max(input_backoff_linear, 1e-12))
    normalized_amplitude = np.abs(signal) / saturation_amplitude
    phase = np.angle(signal)
    smoothness = 3.0
    am_am_gain = (1.0 + normalized_amplitude ** (2.0 * smoothness)) ** (
        -1.0 / (2.0 * smoothness)
    )
    max_am_pm_rad = np.deg2rad(25.0)
    am_pm = max_am_pm_rad * normalized_amplitude**4 / (1.0 + normalized_amplitude**4)
    output = np.abs(signal) * am_am_gain * np.exp(1j * (phase + am_pm))

    probe = np.linspace(0.0, 2.0 * saturation_amplitude, 240)
    probe_normalized = probe / saturation_amplitude
    probe_am = probe * (1.0 + probe_normalized ** (2.0 * smoothness)) ** (
        -1.0 / (2.0 * smoothness)
    )
    probe_pm = max_am_pm_rad * probe_normalized**4 / (1.0 + probe_normalized**4)
    return output, probe, probe_am, np.rad2deg(probe_pm)


def apply_digital_predistortion(
    signal: np.ndarray, backoff_db: float, strength: float = 0.75
) -> np.ndarray:
    """Apply a bounded inverse-like predistorter for the educational HPA model."""

    saturation_amplitude = np.sqrt(max(db_to_linear(backoff_db), 1e-12))
    normalized_amplitude = np.abs(signal) / saturation_amplitude
    # Approximate inverse of the smooth AM/AM compression, limited so it cannot
    # unrealistically create unbounded drive near saturation.
    expansion = (1.0 + normalized_amplitude**6) ** (1.0 / 6.0)
    expansion = 1.0 + strength * (np.minimum(expansion, 1.8) - 1.0)
    phase_lead = (
        -strength
        * np.deg2rad(25.0)
        * normalized_amplitude**4
        / (1.0 + normalized_amplitude**4)
    )
    return signal * expansion * np.exp(1j * phase_lead)


def apply_iq_impairment(signal: np.ndarray, impairment: str) -> np.ndarray:
    """Apply one named I/Q impairment used by the standalone demo."""

    i = np.real(signal).copy()
    q = np.imag(signal).copy()
    if impairment == "amplitude_imbalance_3db":
        i *= db_to_linear(3.0 / 2.0)
        q /= db_to_linear(3.0 / 2.0)
    elif impairment == "phase_imbalance_20deg":
        q_complex = 1j * q * np.exp(1j * np.deg2rad(20.0))
        return i + q_complex
    elif impairment == "i_dc_offset_1e-8":
        i += 1e-8
    elif impairment == "q_dc_offset_5e-8":
        q += 5e-8
    return i + 1j * q


def correct_iq_imbalance(signal: np.ndarray) -> np.ndarray:
    """Normalize I/Q branch means and standard deviations as a simple correction."""

    i = np.real(signal) - np.mean(np.real(signal))
    q = np.imag(signal) - np.mean(np.imag(signal))
    i_std = np.std(i) or 1.0
    q_std = np.std(q) or 1.0
    target = 0.5 * (i_std + q_std)
    return (i / i_std * target) + 1j * (q / q_std * target)


def agc_align_to_reference(
    rx: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, complex]:
    """Apply one-tap complex AGC/carrier alignment using the known demo frame."""

    denominator = np.vdot(rx, rx)
    if abs(denominator) <= 1e-300:
        return rx.copy(), 1.0 + 0.0j
    alpha = complex(np.vdot(reference, rx) / denominator)
    return alpha * rx, alpha


def calculate_evm_metrics(
    rx: np.ndarray, reference: np.ndarray
) -> tuple[float, float, np.ndarray, complex]:
    """Calculate RMS EVM and EVM-derived SNR after optimal gain/phase alignment."""

    rx_aligned, alpha = agc_align_to_reference(rx, reference)
    evm_rms = float(
        np.sqrt(
            np.mean(np.abs(rx_aligned - reference) ** 2)
            / max(np.mean(np.abs(reference) ** 2), 1e-300)
        )
    )
    evm_percent = 100.0 * evm_rms
    evm_snr_db = float("inf") if evm_rms == 0.0 else -20.0 * np.log10(evm_rms)
    return evm_percent, evm_snr_db, rx_aligned, alpha


def _rf_link_budget(config: RFSatelliteLinkDemoConfig) -> dict[str, float]:
    frequency_hz = config.frequency_mhz * 1e6
    range_m = config.altitude_km * 1_000.0
    tx_gain_dbi = parabolic_dish_gain_dbi(config.tx_dish_diameter_m, frequency_hz)
    rx_gain_dbi = parabolic_dish_gain_dbi(config.rx_dish_diameter_m, frequency_hz)
    path_loss_db = free_space_path_loss_db(range_m, frequency_hz)
    tx_power_dbw = 10.0
    received_power_dbw = tx_power_dbw + tx_gain_dbi + rx_gain_dbi - path_loss_db
    symbol_rate_hz = 250e3
    noise_power_w = (
        0.0
        if config.noise_temperature_k == 0.0
        else BOLTZMANN_W_PER_HZ_K * config.noise_temperature_k * symbol_rate_hz
    )
    if noise_power_w == 0.0:
        link_budget_snr_db = float("inf")
        ebno_db = float("inf")
    else:
        link_budget_snr_db = received_power_dbw - linear_to_db(noise_power_w)
        ebno_db = link_budget_snr_db - linear_to_db(_BITS_PER_QAM16_SYMBOL)
    return {
        "frequency_hz": frequency_hz,
        "range_m": range_m,
        "tx_power_dbw": tx_power_dbw,
        "tx_gain_dbi": tx_gain_dbi,
        "rx_gain_dbi": rx_gain_dbi,
        "path_loss_db": path_loss_db,
        "received_power_dbw": received_power_dbw,
        "symbol_rate_hz": symbol_rate_hz,
        "sample_rate_hz": 1.0e6,
        "link_budget_snr_db": link_budget_snr_db,
        "ebno_db": ebno_db,
    }


def _simulate_rf_baseband_step(
    rng: np.random.Generator,
    config: RFSatelliteLinkDemoConfig,
    link_budget_snr_db: float,
    num_bits: int = 16_384,
) -> RFBasebandStepResult:
    """Run one symbol-rate 16-QAM RF chain with honest scaling/noise accounting."""

    if num_bits % _BITS_PER_QAM16_SYMBOL:
        raise ValueError("num_bits must be a multiple of 4")

    bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)
    symbols = qam16_modulate(bits)
    hpa_input = (
        apply_digital_predistortion(symbols, config.hpa_backoff_db)
        if config.digital_predistortion
        else symbols
    )
    hpa_symbols, probe_in, probe_am, probe_pm_deg = apply_hpa_nonlinearity(
        hpa_input, config.hpa_backoff_db
    )

    received_noiseless = hpa_symbols.copy()
    symbol_times = np.arange(symbols.size, dtype=float) / 250e3
    if config.doppler_error_hz:
        received_noiseless *= np.exp(
            1j * 2.0 * np.pi * config.doppler_error_hz * symbol_times
        )
    phase_noise_std = _PHASE_NOISE_STD_RAD[config.phase_noise_level]
    if phase_noise_std:
        received_noiseless *= np.exp(
            1j * rng.normal(0.0, phase_noise_std, symbols.size)
        )
    received_noiseless = apply_iq_impairment(received_noiseless, config.iq_impairment)

    if np.isinf(link_budget_snr_db):
        noise = np.zeros_like(received_noiseless)
        measured_baseband_snr_db = float("inf")
    else:
        snr_linear = db_to_linear(link_budget_snr_db)
        signal_power = float(np.mean(np.abs(received_noiseless) ** 2))
        complex_noise_variance = signal_power / snr_linear
        noise = np.sqrt(complex_noise_variance / 2.0) * (
            rng.normal(size=symbols.size) + 1j * rng.normal(size=symbols.size)
        )
        measured_baseband_snr_db = linear_to_db(
            signal_power / max(float(np.mean(np.abs(noise) ** 2)), 1e-300)
        )

    received = received_noiseless + noise
    if config.doppler_correction and config.doppler_error_hz:
        received *= np.exp(-1j * 2.0 * np.pi * config.doppler_error_hz * symbol_times)
    if config.dc_offset_correction:
        received -= np.mean(received)
    if config.iq_correction:
        received = correct_iq_imbalance(received)

    evm_percent, evm_snr_db, received_agc, agc_alpha = calculate_evm_metrics(
        received, symbols
    )
    demodulated_bits = qam16_demodulate(received_agc)
    bit_errors = int(np.count_nonzero(bits != demodulated_bits))
    ber = bit_errors / bits.size

    hpa_symbols_agc, _ = agc_align_to_reference(hpa_symbols, symbols)
    return RFBasebandStepResult(
        bits=bits,
        symbols=symbols,
        hpa_symbols=hpa_symbols_agc,
        received_symbols=received,
        received_symbols_agc=received_agc,
        noise_symbols=noise,
        bit_errors=bit_errors,
        ber=ber,
        evm_percent=evm_percent,
        evm_snr_db=evm_snr_db,
        measured_baseband_snr_db=measured_baseband_snr_db,
        agc_alpha=agc_alpha,
        hpa_probe_in=probe_in,
        hpa_probe_am=probe_am,
        hpa_probe_pm_deg=probe_pm_deg,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_db(value: float) -> str:
    return "inf" if np.isinf(value) else f"{value:.2f} dB"


def _save_rf_plots(
    output_dir: Path, rows: list[dict[str, object]], artifacts: dict[str, np.ndarray]
) -> None:
    """Save the two requested standalone RF figures."""

    import matplotlib.pyplot as plt

    for obsolete_plot in (
        "received_constellation.png",
        "hpa_am_am.png",
        "hpa_am_pm.png",
        "ber_vs_step.png",
        "evm_vs_step.png",
        "link_budget_bar.png",
    ):
        (output_dir / obsolete_plot).unlink(missing_ok=True)

    def savefig(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=160)
        plt.close()

    fs = float(artifacts["sample_rate_hz"])
    tx = artifacts["tx_waveform"]
    rx = artifacts["rx_waveform"]
    freq = np.fft.fftshift(np.fft.fftfreq(tx.size, d=1.0 / fs)) / 1e3
    tx_psd = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(tx))) + 1e-15)
    rx_psd = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(rx))) + 1e-15)
    plt.figure(figsize=(9, 5))
    plt.plot(freq, tx_psd, color="gold", label="Transmitted 16-QAM waveform")
    plt.plot(freq, rx_psd, color="royalblue", label="Received after AGC/corrections")
    plt.title("RF Satellite Link Power Spectrum")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Power (dB, normalized)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    savefig("power_spectrum.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].scatter(
        np.real(artifacts["symbols"]),
        np.imag(artifacts["symbols"]),
        s=10,
        alpha=0.45,
        label="Before HPA",
    )
    axes[0].scatter(
        np.real(artifacts["hpa_symbols"]),
        np.imag(artifacts["hpa_symbols"]),
        s=10,
        alpha=0.45,
        label="After HPA (AGC aligned)",
    )
    axes[0].set_title("16-QAM Before and After HPA")
    axes[0].set_xlabel("In-phase")
    axes[0].set_ylabel("Quadrature")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    axes[1].scatter(
        np.real(artifacts["received_symbols_agc"]),
        np.imag(artifacts["received_symbols_agc"]),
        s=9,
        alpha=0.35,
        label="Received after correction + AGC",
    )
    reference = qam16_modulate(
        np.array(
            [
                [a, b, c, d]
                for a in (0, 1)
                for b in (0, 1)
                for c in (0, 1)
                for d in (0, 1)
            ],
            dtype=np.uint8,
        ).reshape(-1)
    )
    axes[1].scatter(
        np.real(reference),
        np.imag(reference),
        s=55,
        marker="x",
        color="black",
        label="Reference 16-QAM",
    )
    axes[1].set_title("Received Constellation Used for Demodulation")
    axes[1].set_xlabel("In-phase")
    axes[1].set_ylabel("Quadrature")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend()
    fig.suptitle("RF Constellations")
    savefig("constellation_before_after_hpa.png")


def run_rf_satellite_link_demo(
    config: RFSatelliteLinkDemoConfig | None = None,
) -> dict[str, object]:
    """Run a standalone RF satellite link demonstration and save artifacts."""

    config = config or RFSatelliteLinkDemoConfig()
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    budget = _rf_link_budget(config)
    rows: list[dict[str, object]] = []
    total_errors = 0
    total_bits = 0
    artifacts: dict[str, np.ndarray] = {}

    for step in range(config.steps):
        result = _simulate_rf_baseband_step(rng, config, budget["link_budget_snr_db"])
        total_errors += result.bit_errors
        total_bits += result.bits.size
        row = {
            "step": step,
            "altitude_km": config.altitude_km,
            "frequency_mhz": config.frequency_mhz,
            "range_m": budget["range_m"],
            "tx_dish_diameter_m": config.tx_dish_diameter_m,
            "rx_dish_diameter_m": config.rx_dish_diameter_m,
            "tx_gain_dbi": budget["tx_gain_dbi"],
            "rx_gain_dbi": budget["rx_gain_dbi"],
            "path_loss_db": budget["path_loss_db"],
            "received_power_dbw": budget["received_power_dbw"],
            "noise_temperature_k": config.noise_temperature_k,
            "hpa_backoff_db": config.hpa_backoff_db,
            "doppler_error_hz": config.doppler_error_hz,
            "phase_noise_level": config.phase_noise_level,
            "iq_impairment": config.iq_impairment,
            "digital_predistortion_enabled": config.digital_predistortion,
            "dc_offset_correction_enabled": config.dc_offset_correction,
            "doppler_correction_enabled": config.doppler_correction,
            "iq_correction_enabled": config.iq_correction,
            "evm_percent": result.evm_percent,
            "evm_snr_db": result.evm_snr_db,
            "ber": result.ber,
            "bit_errors": result.bit_errors,
            "num_bits": int(result.bits.size),
            # Keep the original column while making its meaning explicit in new columns.
            "snr_db": budget["link_budget_snr_db"],
            "link_budget_snr_db": budget["link_budget_snr_db"],
            "measured_baseband_snr_db": result.measured_baseband_snr_db,
            "ebno_db": budget["ebno_db"],
        }
        rows.append(row)
        if step == config.steps - 1:
            tx_waveform, _ = educational_pulse_shape(result.symbols)
            rx_waveform, _ = educational_pulse_shape(result.received_symbols_agc)
            artifacts = {
                "sample_rate_hz": np.array(budget["sample_rate_hz"]),
                "tx_waveform": tx_waveform,
                "rx_waveform": rx_waveform,
                "symbols": result.symbols,
                "hpa_symbols": result.hpa_symbols,
                "received_symbols_agc": result.received_symbols_agc,
            }

    _write_csv(output_dir / "rf_link_metrics.csv", rows)
    final_ber = total_errors / total_bits
    mean_link_snr = float(np.mean([float(r["link_budget_snr_db"]) for r in rows]))
    mean_measured_snr = float(
        np.mean([float(r["measured_baseband_snr_db"]) for r in rows])
    )
    mean_evm_snr = float(np.mean([float(r["evm_snr_db"]) for r in rows]))
    summary = {
        "steps": config.steps,
        "total_bits": total_bits,
        "total_bit_errors": total_errors,
        "final_ber": final_ber,
        "mean_ber": float(np.mean([float(r["ber"]) for r in rows])),
        "mean_evm_percent": float(np.mean([float(r["evm_percent"]) for r in rows])),
        "mean_snr_db": mean_link_snr,
        "mean_link_budget_snr_db": mean_link_snr,
        "mean_measured_baseband_snr_db": mean_measured_snr,
        "mean_evm_snr_db": mean_evm_snr,
        "mean_ebno_db": float(np.mean([float(r["ebno_db"]) for r in rows])),
        "tx_gain_dbi": budget["tx_gain_dbi"],
        "rx_gain_dbi": budget["rx_gain_dbi"],
        "path_loss_db": budget["path_loss_db"],
        "mean_received_power_dbw": float(
            np.mean([float(r["received_power_dbw"]) for r in rows])
        ),
        "configuration_used": {**config.__dict__, "output_dir": str(output_dir)},
    }
    with (output_dir / "rf_link_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    if config.make_plots:
        _save_rf_plots(output_dir, rows, artifacts)

    print("\nRF Satellite Link Demo Summary")
    print("=" * 34)
    print(f"Output folder: {output_dir}")
    print(
        f"Configuration: altitude={config.altitude_km:g} km, frequency={config.frequency_mhz:g} MHz, noise={config.noise_temperature_k:g} K"
    )
    print(f"Path loss: {budget['path_loss_db']:.2f} dB")
    print(
        f"Tx/Rx antenna gains: {budget['tx_gain_dbi']:.2f} / {budget['rx_gain_dbi']:.2f} dBi"
    )
    print(f"Received power: {summary['mean_received_power_dbw']:.2f} dBW")
    print(f"Link-budget SNR: {_format_db(summary['mean_link_budget_snr_db'])}")
    print(
        f"Measured baseband SNR: {_format_db(summary['mean_measured_baseband_snr_db'])}"
    )
    print(f"Eb/No: {_format_db(summary['mean_ebno_db'])}")
    print(f"EVM: {summary['mean_evm_percent']:.3f}%")
    print(f"EVM-derived SNR: {_format_db(summary['mean_evm_snr_db'])}")
    print(
        f"BER: {summary['final_ber']:.6g} ({summary['total_bit_errors']} / {summary['total_bits']} bit errors)"
    )
    if (
        np.isfinite(summary["mean_measured_baseband_snr_db"])
        and np.isfinite(summary["mean_evm_snr_db"])
        and abs(summary["mean_measured_baseband_snr_db"] - summary["mean_evm_snr_db"])
        > 3.0
    ):
        print(
            "WARNING: measured baseband SNR and EVM-derived SNR differ by more than 3 dB; distortion/impairments dominate the EVM."
        )
    print(
        "Impairments/corrections: "
        f"HPA backoff={config.hpa_backoff_db:g} dB, Doppler={config.doppler_error_hz:g} Hz, "
        f"phase noise={config.phase_noise_level}, IQ={config.iq_impairment}, "
        f"DPD={config.digital_predistortion}, DC correction={config.dc_offset_correction}, "
        f"Doppler correction={config.doppler_correction}, IQ correction={config.iq_correction}"
    )
    return summary


def _validation_line(name: str, passed: bool, details: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name} - {details}")
    return passed


def run_rf_validation() -> bool:
    """Run internal consistency checks for the standalone RF baseband chain."""

    print("\nRF Satellite Link Validation")
    print("=" * 34)
    all_passed = True
    rng = np.random.default_rng(12345)

    bits = rng.integers(0, 2, size=4096, dtype=np.uint8)
    symbols = qam16_modulate(bits)
    decoded = qam16_demodulate(symbols)
    ber = np.count_nonzero(bits != decoded) / bits.size
    avg_power = float(np.mean(np.abs(symbols) ** 2))
    all_passed &= _validation_line(
        "Test A: QAM mapper/demapper no noise",
        ber == 0.0 and abs(avg_power - 1.0) < 0.03,
        f"BER={ber:.3g}, average symbol power={avg_power:.6f}",
    )

    no_noise_config = RFSatelliteLinkDemoConfig(noise_temperature_k=0.0)
    no_noise_result = _simulate_rf_baseband_step(
        rng, no_noise_config, float("inf"), num_bits=4096
    )
    all_passed &= _validation_line(
        "Test B: End-to-end baseband no impairments, no noise",
        no_noise_result.ber == 0.0 and no_noise_result.evm_percent < 0.1,
        f"BER={no_noise_result.ber:.3g}, EVM={no_noise_result.evm_percent:.5f}%",
    )

    default_config = RFSatelliteLinkDemoConfig(steps=3, make_plots=False)
    default_budget = _rf_link_budget(default_config)
    default_errors = 0
    default_bits = 0
    default_evms = []
    default_evm_snrs = []
    default_measured_snrs = []
    for _ in range(default_config.steps):
        result = _simulate_rf_baseband_step(
            rng, default_config, default_budget["link_budget_snr_db"], num_bits=16_384
        )
        default_errors += result.bit_errors
        default_bits += result.bits.size
        default_evms.append(result.evm_percent)
        default_evm_snrs.append(result.evm_snr_db)
        default_measured_snrs.append(result.measured_baseband_snr_db)
    default_ber = default_errors / default_bits
    default_evm_snr = float(np.mean(default_evm_snrs))
    default_measured_snr = float(np.mean(default_measured_snrs))
    all_passed &= _validation_line(
        "Test C: Default MATLAB-like case",
        195.0 < default_budget["path_loss_db"] < 196.0
        and 22.0 < default_budget["tx_gain_dbi"] < 22.6
        and default_ber < 1e-3
        and abs(default_evm_snr - default_measured_snr) < 1.5,
        "path_loss="
        f"{default_budget['path_loss_db']:.2f} dB, gains={default_budget['tx_gain_dbi']:.2f}/{default_budget['rx_gain_dbi']:.2f} dBi, "
        f"BER={default_ber:.3g}, EVM={np.mean(default_evms):.3f}%, "
        f"measured SNR={default_measured_snr:.2f} dB, EVM SNR={default_evm_snr:.2f} dB",
    )

    zero_noise_config = RFSatelliteLinkDemoConfig(
        noise_temperature_k=0.0, make_plots=False
    )
    zero_noise_result = _simulate_rf_baseband_step(
        rng, zero_noise_config, float("inf"), num_bits=16_384
    )
    all_passed &= _validation_line(
        "Test D: Noise temperature = 0 K",
        zero_noise_result.ber == 0.0,
        f"BER={zero_noise_result.ber:.3g}, bit_errors={zero_noise_result.bit_errors}",
    )

    severe_config = RFSatelliteLinkDemoConfig(
        hpa_backoff_db=1.0,
        phase_noise_level="high",
        iq_impairment="amplitude_imbalance_3db",
        doppler_error_hz=3.0,
        make_plots=False,
    )
    severe_budget = _rf_link_budget(severe_config)
    severe_result = _simulate_rf_baseband_step(
        rng, severe_config, severe_budget["link_budget_snr_db"], num_bits=16_384
    )
    all_passed &= _validation_line(
        "Test E: Severe impairment case",
        severe_result.ber > default_ber
        and severe_result.evm_percent > float(np.mean(default_evms)) * 1.5,
        f"BER={severe_result.ber:.3g}, EVM={severe_result.evm_percent:.2f}%",
    )

    print("=" * 34)
    print("Overall validation: " + ("PASS" if all_passed else "FAIL"))
    return bool(all_passed)


def _build_rf_demo_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standalone RF satellite link demo."
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/rf_satellite_link_demo")
    )
    parser.add_argument("--altitude-km", type=float, default=35_600.0)
    parser.add_argument("--frequency-mhz", type=float, default=4_000.0)
    parser.add_argument("--tx-dish-diameter-m", type=float, default=0.4)
    parser.add_argument("--rx-dish-diameter-m", type=float, default=0.4)
    parser.add_argument(
        "--noise-temperature-k",
        type=float,
        choices=[0.0, 20.0, 290.0, 500.0],
        default=20.0,
    )
    parser.add_argument(
        "--hpa-backoff-db", type=float, choices=[30.0, 7.0, 1.0], default=30.0
    )
    parser.add_argument(
        "--doppler-error-hz", type=float, choices=[0.0, 3.0], default=0.0
    )
    parser.add_argument(
        "--phase-noise-level", choices=list(_PHASE_NOISE_STD_RAD), default="negligible"
    )
    parser.add_argument(
        "--iq-impairment", choices=_SUPPORTED_IQ_IMPAIRMENTS, default="none"
    )
    parser.add_argument("--digital-predistortion", action="store_true")
    parser.add_argument("--dc-offset-correction", action="store_true")
    parser.add_argument("--doppler-correction", action="store_true")
    parser.add_argument("--iq-correction", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--validation", action="store_true")
    return parser


def _main() -> None:
    args = _build_rf_demo_arg_parser().parse_args()
    if args.validation:
        run_rf_validation()
        return
    run_rf_satellite_link_demo(
        RFSatelliteLinkDemoConfig(
            steps=args.steps,
            seed=args.seed,
            output_dir=args.output_dir,
            altitude_km=args.altitude_km,
            frequency_mhz=args.frequency_mhz,
            tx_dish_diameter_m=args.tx_dish_diameter_m,
            rx_dish_diameter_m=args.rx_dish_diameter_m,
            noise_temperature_k=args.noise_temperature_k,
            hpa_backoff_db=args.hpa_backoff_db,
            doppler_error_hz=args.doppler_error_hz,
            phase_noise_level=args.phase_noise_level,
            iq_impairment=args.iq_impairment,
            digital_predistortion=args.digital_predistortion,
            dc_offset_correction=args.dc_offset_correction,
            doppler_correction=args.doppler_correction,
            iq_correction=args.iq_correction,
            make_plots=not args.no_plots,
        )
    )


if __name__ == "__main__":
    _main()
