"""30 kHz CW narrowband DOA simulation using the reusable CBF module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .cbf import CBFResult, estimate_doa_cbf, steering_vector
except ImportError:  # Allows running this file directly: python code/signaldoa/cw_doa.py
    from cbf import CBFResult, estimate_doa_cbf, steering_vector


@dataclass(frozen=True)
class SimulationResult:
    """Container for simulated ULA narrowband CW data."""

    X: np.ndarray
    X_clean: np.ndarray
    signal: np.ndarray
    time: np.ndarray
    positions: np.ndarray
    theta_true_deg: float
    c: float
    fc: float
    wavelength: float
    spacing: float
    snr_db: float


def make_ula_positions(num_sensors: int, spacing: float) -> np.ndarray:
    """Return ULA element positions along one axis."""
    if num_sensors < 1:
        raise ValueError("num_sensors must be at least 1.")
    if spacing <= 0:
        raise ValueError("spacing must be positive.")
    return np.arange(num_sensors, dtype=float) * spacing


def add_complex_awgn(
    signal_matrix: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add circular complex white Gaussian noise at the requested SNR."""
    signal_power = np.mean(np.abs(signal_matrix) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(signal_matrix.shape) + 1j * rng.standard_normal(signal_matrix.shape)
    )
    return signal_matrix + noise


def simulate_cw_ula(
    theta_deg: float = 25.0,
    fc: float = 30e3,
    c: float = 1500.0,
    num_sensors: int = 16,
    spacing: float | None = None,
    fs: float = 240e3,
    duration: float = 0.020,
    snr_db: float = 10.0,
    seed: int | None = 20260602,
) -> SimulationResult:
    """Simulate noisy far-field narrowband CW snapshots received by a ULA."""
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if duration <= 0:
        raise ValueError("duration must be positive.")

    wavelength = c / fc
    if spacing is None:
        spacing = wavelength / 2

    rng = np.random.default_rng(seed)
    positions = make_ula_positions(num_sensors=num_sensors, spacing=spacing)
    time = np.arange(0, duration, 1 / fs)

    signal = np.exp(1j * 2 * np.pi * fc * time)
    a_true = steering_vector(theta_deg=theta_deg, fc=fc, c=c, positions=positions)
    X_clean = a_true[:, None] * signal[None, :]
    X = add_complex_awgn(X_clean, snr_db=snr_db, rng=rng)

    return SimulationResult(
        X=X,
        X_clean=X_clean,
        signal=signal,
        time=time,
        positions=positions,
        theta_true_deg=theta_deg,
        c=c,
        fc=fc,
        wavelength=wavelength,
        spacing=spacing,
        snr_db=snr_db,
    )


def run_demo(plot: bool = True) -> tuple[SimulationResult, CBFResult]:
    """Run the default 30 kHz CW + 16-sensor ULA + CBF demo."""
    sim = simulate_cw_ula()
    scan_angles = np.linspace(-90, 90, 1801)
    cbf = estimate_doa_cbf(
        X=sim.X,
        scan_angles_deg=scan_angles,
        fc=sim.fc,
        c=sim.c,
        positions=sim.positions,
    )

    print(f"fc = {sim.fc / 1e3:.1f} kHz")
    print(f"lambda = {sim.wavelength:.4f} m")
    print(f"M = {len(sim.positions)}, d = {sim.spacing:.4f} m")
    print(f"X shape = {sim.X.shape}  # (sensors, snapshots)")
    print(f"true DOA = {sim.theta_true_deg:.2f} deg")
    print(f"CBF estimated DOA = {cbf.theta_hat_deg:.2f} deg")
    print(f"absolute error = {abs(cbf.theta_hat_deg - sim.theta_true_deg):.2f} deg")

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(cbf.scan_angles_deg, cbf.power_db, linewidth=2, label="CBF spectrum")
        plt.axvline(
            sim.theta_true_deg,
            color="tab:green",
            linestyle="--",
            linewidth=2,
            label=f"true {sim.theta_true_deg:.1f} deg",
        )
        plt.axvline(
            cbf.theta_hat_deg,
            color="tab:red",
            linestyle=":",
            linewidth=2.5,
            label=f"estimated {cbf.theta_hat_deg:.1f} deg",
        )
        plt.ylim(-25, 3)
        plt.xlim(-90, 90)
        plt.xlabel("Scan angle (deg)")
        plt.ylabel("Normalized power (dB)")
        plt.title("Conventional Beamforming DOA Spectrum")
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    return sim, cbf


if __name__ == "__main__":
    run_demo(plot=True)
