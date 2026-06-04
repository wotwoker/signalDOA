"""MUSIC utilities for narrowband DOA estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .cbf import steering_matrix
except ImportError:
    from cbf import steering_matrix


@dataclass(frozen=True)
class MUSICResult:
    """Container for MUSIC DOA estimation results."""

    theta_hat_deg: float
    scan_angles_deg: np.ndarray
    power: np.ndarray
    power_db: np.ndarray
    eigvals: np.ndarray


def music_spectrum(
    X: np.ndarray,
    scan_angles_deg: np.ndarray,
    fc: float,
    c: float,
    positions: np.ndarray,
    num_sources: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the MUSIC spatial spectrum.

    MUSIC separates the sample covariance matrix into a signal subspace and a
    noise subspace. True steering vectors are nearly orthogonal to the noise
    subspace, so 1 / ||E_n^H a(theta)||^2 peaks at source directions.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array with shape (num_sensors, num_snapshots).")
    if X.shape[0] != len(positions):
        raise ValueError("X.shape[0] must match the number of sensor positions.")
    if not 0 < num_sources < len(positions):
        raise ValueError("num_sources must be between 1 and num_sensors - 1.")

    num_sensors = len(positions)
    Rxx = X @ X.conj().T / X.shape[1]

    eigvals, eigvecs = np.linalg.eigh(Rxx)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    En = eigvecs[:, num_sources:num_sensors]
    A = steering_matrix(scan_angles_deg=scan_angles_deg, fc=fc, c=c, positions=positions)

    EnH_A = En.conj().T @ A
    denominator = np.sum(np.abs(EnH_A) ** 2, axis=0)
    power = 1.0 / np.maximum(denominator, np.finfo(float).tiny)
    return power, eigvals


def estimate_doa_music(
    X: np.ndarray,
    scan_angles_deg: np.ndarray,
    fc: float,
    c: float,
    positions: np.ndarray,
    num_sources: int,
) -> MUSICResult:
    """Estimate the strongest DOA using MUSIC."""
    power, eigvals = music_spectrum(
        X=X,
        scan_angles_deg=scan_angles_deg,
        fc=fc,
        c=c,
        positions=positions,
        num_sources=num_sources,
    )
    theta_hat_deg = float(scan_angles_deg[np.argmax(power)])
    power_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny) / np.max(power))
    return MUSICResult(
        theta_hat_deg=theta_hat_deg,
        scan_angles_deg=scan_angles_deg,
        power=power,
        power_db=power_db,
        eigvals=eigvals,
    )
