"""MVDR/Capon beamforming utilities for narrowband DOA estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .cbf import steering_matrix
except ImportError:
    from cbf import steering_matrix


@dataclass(frozen=True)
class MVDRResult:
    """Container for MVDR DOA estimation results."""

    theta_hat_deg: float
    scan_angles_deg: np.ndarray
    power: np.ndarray
    power_db: np.ndarray


def mvdr_spectrum(
    X: np.ndarray,
    scan_angles_deg: np.ndarray,
    fc: float,
    c: float,
    positions: np.ndarray,
    diagonal_loading: float = 1e-3,
) -> np.ndarray:
    """Compute the MVDR/Capon spatial spectrum.

    MVDR solves:
        min_w w^H Rxx w, subject to w^H a(theta) = 1

    Its spectrum is:
        P(theta) = 1 / (a(theta)^H Rxx^{-1} a(theta))

    A small diagonal loading term improves numerical stability when Rxx is
    ill-conditioned, especially for coherent CW sources.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array with shape (num_sensors, num_snapshots).")
    if X.shape[0] != len(positions):
        raise ValueError("X.shape[0] must match the number of sensor positions.")

    num_sensors = len(positions)
    Rxx = X @ X.conj().T / X.shape[1]

    loading = diagonal_loading * np.trace(Rxx).real / num_sensors
    R_loaded = Rxx + loading * np.eye(num_sensors)

    A = steering_matrix(scan_angles_deg=scan_angles_deg, fc=fc, c=c, positions=positions)

    # Use solve instead of explicitly computing inv(R_loaded).
    R_inv_A = np.linalg.solve(R_loaded, A)
    denominator = np.real(np.sum(A.conj() * R_inv_A, axis=0))
    power = 1.0 / np.maximum(denominator, np.finfo(float).tiny)
    return power


def estimate_doa_mvdr(
    X: np.ndarray,
    scan_angles_deg: np.ndarray,
    fc: float,
    c: float,
    positions: np.ndarray,
    diagonal_loading: float = 1e-3,
) -> MVDRResult:
    """Estimate the strongest DOA using MVDR/Capon beamforming."""
    power = mvdr_spectrum(
        X=X,
        scan_angles_deg=scan_angles_deg,
        fc=fc,
        c=c,
        positions=positions,
        diagonal_loading=diagonal_loading,
    )
    theta_hat_deg = float(scan_angles_deg[np.argmax(power)])
    power_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny) / np.max(power))
    return MVDRResult(
        theta_hat_deg=theta_hat_deg,
        scan_angles_deg=scan_angles_deg,
        power=power,
        power_db=power_db,
    )
