"""Conventional beamforming utilities for narrowband DOA estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CBFResult:
    """Container for conventional beamforming DOA results."""

    theta_hat_deg: float
    scan_angles_deg: np.ndarray
    power: np.ndarray
    power_db: np.ndarray


def steering_vector(
    theta_deg: float,
    fc: float,
    c: float,
    positions: np.ndarray,
) -> np.ndarray:
    """Return the far-field narrowband ULA steering vector.

    Angle is measured relative to array broadside. The phase convention matches
    X = a(theta) s(t) + noise.
    """
    theta_rad = np.deg2rad(theta_deg)
    tau = positions * np.sin(theta_rad) / c
    return np.exp(-1j * 2 * np.pi * fc * tau)


def steering_matrix(
    scan_angles_deg: np.ndarray,
    fc: float,
    c: float,
    positions: np.ndarray,
) -> np.ndarray:
    """Return steering vectors stacked column-wise for all scan angles."""
    return np.column_stack(
        [steering_vector(theta, fc=fc, c=c, positions=positions) for theta in scan_angles_deg]
    )


def cbf_spectrum(
    X: np.ndarray,
    scan_angles_deg: np.ndarray,
    fc: float,
    c: float,
    positions: np.ndarray,
) -> np.ndarray:
    """Compute the conventional beamforming spatial spectrum.

    X must have shape (num_sensors, num_snapshots).
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array with shape (num_sensors, num_snapshots).")
    if X.shape[0] != len(positions):
        raise ValueError("X.shape[0] must match the number of sensor positions.")

    Rxx = X @ X.conj().T / X.shape[1]
    A = steering_matrix(scan_angles_deg=scan_angles_deg, fc=fc, c=c, positions=positions)
    RA = Rxx @ A
    power = np.real(np.sum(A.conj() * RA, axis=0)) / (len(positions) ** 2)
    return power


def estimate_doa_cbf(
    X: np.ndarray,
    scan_angles_deg: np.ndarray,
    fc: float,
    c: float,
    positions: np.ndarray,
) -> CBFResult:
    """Estimate DOA using conventional beamforming."""
    power = cbf_spectrum(
        X=X,
        scan_angles_deg=scan_angles_deg,
        fc=fc,
        c=c,
        positions=positions,
    )
    theta_hat_deg = float(scan_angles_deg[np.argmax(power)])
    power_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny) / np.max(power))
    return CBFResult(
        theta_hat_deg=theta_hat_deg,
        scan_angles_deg=scan_angles_deg,
        power=power,
        power_db=power_db,
    )
