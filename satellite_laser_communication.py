"""Satellite laser communication link-budget and pointing simulation.

The model follows the style of open satellite-to-ground laser communication
calculators: it computes optical free-space loss, atmospheric attenuation,
pointing loss, receiver aperture gain, background/thermal noise, and resulting
SNR/capacity for a laser downlink. It also plugs into this repository's
satellite clock objects so laser ranging/time-transfer measurements can be used
as synchronization corrections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.satellite import Satellite, SatelliteState

PLANCK_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


def db_to_linear(value_db: float) -> float:
    """Convert a decibel value to linear scale."""

    return 10.0 ** (value_db / 10.0)


def linear_to_db(value: float, floor: float = 1e-300) -> float:
    """Convert a linear value to decibels with a numerical floor."""

    return 10.0 * np.log10(max(float(value), floor))


def optical_free_space_loss_db(range_m: float, wavelength_m: float) -> float:
    """Return optical free-space path loss."""

    if range_m <= 0.0:
        raise ValueError("range_m must be positive")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")
    return linear_to_db((4.0 * np.pi * range_m / wavelength_m) ** 2)


def diffraction_limited_divergence_rad(wavelength_m: float, aperture_diameter_m: float) -> float:
    """Approximate full-angle diffraction-limited beam divergence."""

    if aperture_diameter_m <= 0.0:
        raise ValueError("aperture_diameter_m must be positive")
    return 1.22 * wavelength_m / aperture_diameter_m


def aperture_gain_db(aperture_diameter_m: float, wavelength_m: float, efficiency: float) -> float:
    """Optical aperture gain for a circular telescope."""

    if not 0.0 < efficiency <= 1.0:
        raise ValueError("efficiency must be in (0, 1]")
    return linear_to_db(efficiency * (np.pi * aperture_diameter_m / wavelength_m) ** 2)


def atmospheric_transmittance(elevation_deg: float, zenith_transmittance: float) -> float:
    """Estimate slant-path atmospheric transmittance from zenith clarity."""

    if not 0.0 < zenith_transmittance <= 1.0:
        raise ValueError("zenith_transmittance must be in (0, 1]")
    elevation_rad = np.deg2rad(np.clip(elevation_deg, 1.0, 90.0))
    air_mass = 1.0 / np.sin(elevation_rad)
    return float(zenith_transmittance**air_mass)


def pointing_loss_db(pointing_error_rad: float, beam_divergence_rad: float) -> float:
    """Gaussian beam pointing loss from angular tracking error."""

    if beam_divergence_rad <= 0.0:
        raise ValueError("beam_divergence_rad must be positive")
    loss_linear = np.exp(-2.0 * (pointing_error_rad / beam_divergence_rad) ** 2)
    return linear_to_db(loss_linear)


@dataclass(frozen=True)
class LaserLinkConfig:
    """Configuration for a satellite laser communication terminal."""

    wavelength_m: float = 1_550e-9
    transmit_power_w: float = 2.0
    transmitter_aperture_m: float = 0.08
    receiver_aperture_m: float = 0.40
    optical_efficiency: float = 0.65
    data_rate_bps: float = 1.0e9
    receiver_bandwidth_hz: float = 1.0e9
    detector_quantum_efficiency: float = 0.75
    detector_noise_w: float = 2.0e-12
    background_power_w: float = 5.0e-10
    implementation_loss_db: float = 2.0
    acquisition_snr_threshold_db: float = 3.0
    timing_jitter_floor_s: float = 25.0e-12

    @property
    def beam_divergence_rad(self) -> float:
        """Default diffraction-limited transmit divergence."""

        return diffraction_limited_divergence_rad(self.wavelength_m, self.transmitter_aperture_m)

    @property
    def photon_energy_j(self) -> float:
        """Energy of one photon at the configured wavelength."""

        return PLANCK_J_S * SPEED_OF_LIGHT_M_PER_S / self.wavelength_m


@dataclass(frozen=True)
class LaserLinkGeometry:
    """One satellite laser link geometry snapshot."""

    range_m: float = 800_000.0
    elevation_deg: float = 45.0
    pointing_error_rad: float = 1.0e-6
    zenith_transmittance: float = 0.85


@dataclass(frozen=True)
class LaserLinkMeasurement:
    """Outputs from one satellite laser communication exchange."""

    transmitter_id: int
    receiver_id: int
    received_power_w: float
    received_power_dbw: float
    free_space_loss_db: float
    atmospheric_loss_db: float
    pointing_loss_db: float
    snr_db: float
    photons_per_bit: float
    capacity_bps: float
    propagation_delay_ns: float
    estimated_offset_ns: float
    acquired: bool


@dataclass
class SatelliteLaserCommunication:
    """Laser communication terminal with time-transfer measurements.

    A call to :meth:`measure_downlink` computes the optical link budget and then
    forms a one-way clock-offset estimate, similar in spirit to the sample
    ``SatelliteNode`` time-message code supplied by the user. The measurement can
    immediately drive a bounded synchronization correction on the receiving
    satellite.
    """

    receiver: Satellite
    config: LaserLinkConfig = field(default_factory=LaserLinkConfig)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    measurements: list[LaserLinkMeasurement] = field(default_factory=list)

    def measure_downlink(
        self,
        transmitter: Satellite,
        true_time_s: float,
        geometry: LaserLinkGeometry | None = None,
    ) -> LaserLinkMeasurement:
        """Measure one optical downlink from ``transmitter`` to this receiver."""

        if geometry is None:
            geometry = LaserLinkGeometry()
        if transmitter.state == SatelliteState.INITIALIZING:
            transmitter.complete_initialization()
        if self.receiver.state == SatelliteState.INITIALIZING:
            self.receiver.complete_initialization()

        transmit_time_ns = transmitter.local_time_ns(true_time_s)
        propagation_delay_s = geometry.range_m / SPEED_OF_LIGHT_M_PER_S
        receive_time_ns = self.receiver.local_time_ns(true_time_s + propagation_delay_s)

        fs_loss_db = optical_free_space_loss_db(geometry.range_m, self.config.wavelength_m)
        tx_gain_db = aperture_gain_db(
            self.config.transmitter_aperture_m,
            self.config.wavelength_m,
            self.config.optical_efficiency,
        )
        rx_gain_db = aperture_gain_db(
            self.config.receiver_aperture_m,
            self.config.wavelength_m,
            self.config.optical_efficiency,
        )
        transmittance = atmospheric_transmittance(
            geometry.elevation_deg, geometry.zenith_transmittance
        )
        atmospheric_loss_db = linear_to_db(transmittance)
        beam_pointing_loss_db = pointing_loss_db(
            geometry.pointing_error_rad, self.config.beam_divergence_rad
        )

        received_power_dbw = (
            linear_to_db(self.config.transmit_power_w)
            + tx_gain_db
            + rx_gain_db
            - fs_loss_db
            + atmospheric_loss_db
            + beam_pointing_loss_db
            - self.config.implementation_loss_db
        )
        received_power_w = db_to_linear(received_power_dbw)
        snr_linear = self._snr_linear(received_power_w)
        snr_db = linear_to_db(snr_linear)
        photons_per_bit = received_power_w / (self.config.data_rate_bps * self.config.photon_energy_j)
        capacity_bps = self.config.receiver_bandwidth_hz * np.log2(1.0 + snr_linear)
        acquired = bool(snr_db >= self.config.acquisition_snr_threshold_db)

        estimated_offset_ns = (
            receive_time_ns
            - transmit_time_ns
            - propagation_delay_s * 1e9
            + self._timing_noise_ns(snr_linear)
        )

        measurement = LaserLinkMeasurement(
            transmitter_id=transmitter.node_id,
            receiver_id=self.receiver.node_id,
            received_power_w=received_power_w,
            received_power_dbw=received_power_dbw,
            free_space_loss_db=fs_loss_db,
            atmospheric_loss_db=atmospheric_loss_db,
            pointing_loss_db=beam_pointing_loss_db,
            snr_db=snr_db,
            photons_per_bit=photons_per_bit,
            capacity_bps=float(capacity_bps),
            propagation_delay_ns=propagation_delay_s * 1e9,
            estimated_offset_ns=estimated_offset_ns,
            acquired=acquired,
        )
        self.measurements.append(measurement)
        return measurement

    def apply_last_correction(self, gain: float = 0.7) -> None:
        """Correct the receiver clock using the latest acquired laser link."""

        if not self.measurements:
            return
        measurement = self.measurements[-1]
        if measurement.acquired:
            self.receiver.apply_correction(-gain * measurement.estimated_offset_ns)

    def _snr_linear(self, received_power_w: float) -> float:
        shot_noise_w = np.sqrt(
            2.0 * self.config.photon_energy_j * received_power_w * self.config.receiver_bandwidth_hz
        )
        noise_w = np.hypot(
            self.config.detector_noise_w,
            np.hypot(self.config.background_power_w, shot_noise_w),
        )
        return (received_power_w / max(noise_w, 1e-300)) ** 2

    def _timing_noise_ns(self, snr_linear: float) -> float:
        tracking_std_s = 1.0 / (
            2.0 * np.pi * self.config.receiver_bandwidth_hz * np.sqrt(max(snr_linear, 1e-12))
        )
        std_ns = 1e9 * np.hypot(tracking_std_s, self.config.timing_jitter_floor_s)
        return float(self.rng.normal(0.0, std_ns))


def simulate_laser_communication(
    steps: int = 5,
    seed: int = 17,
    range_m: float = 800_000.0,
    elevation_deg: float = 45.0,
) -> list[LaserLinkMeasurement]:
    """Run a compact laser communication demo with two satellites."""

    rng = np.random.default_rng(seed)
    transmitter = Satellite(node_id=0, offset_ns=125.0, drift_ns_per_s=0.05)
    receiver = Satellite(node_id=1, offset_ns=-420.0, drift_ns_per_s=-0.02)
    terminal = SatelliteLaserCommunication(receiver=receiver, rng=rng)

    results: list[LaserLinkMeasurement] = []
    for step in range(steps):
        pointing_error = abs(rng.normal(1.0e-6, 0.25e-6))
        geometry = LaserLinkGeometry(
            range_m=range_m,
            elevation_deg=elevation_deg,
            pointing_error_rad=pointing_error,
        )
        true_time_s = float(step)
        transmitter.tick(1.0)
        receiver.tick(1.0)
        measurement = terminal.measure_downlink(transmitter, true_time_s, geometry)
        terminal.apply_last_correction(gain=0.7)
        results.append(measurement)
    return results


if __name__ == "__main__":
    for item in simulate_laser_communication():
        print(
            f"tx={item.transmitter_id} rx={item.receiver_id} "
            f"Pr={item.received_power_dbw:.2f} dBW SNR={item.snr_db:.2f} dB "
            f"capacity={item.capacity_bps / 1e9:.2f} Gb/s "
            f"offset={item.estimated_offset_ns:.2f} ns acquired={item.acquired}"
        )
