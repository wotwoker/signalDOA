"""Conventional beamforming utilities for narrowband DOA estimation.""" 

from dataclasses import dataclass  # Lightweight container for result data.

import numpy as np  # Numerical routines and complex math.


@dataclass(frozen=True)  # Immutable result object.
class CBFResult:
    """Container for conventional beamforming DOA results."""  # Data-only holder.

    theta_hat_deg: float  # Estimated DOA (degrees).
    scan_angles_deg: np.ndarray  # Scan grid (degrees).
    power: np.ndarray  # Linear beamformer power across scan grid.
    power_db: np.ndarray  # Normalized power in dB (peak at 0 dB).


def steering_vector(
    theta_deg: float,  # Look direction in degrees.
    fc: float,  # Carrier/center frequency in Hz.
    c: float,  # Propagation speed in m/s.
    positions: np.ndarray,  # Sensor positions along the array axis (meters).
) -> np.ndarray:
    """Return the far-field narrowband ULA steering vector.

    Angle is measured relative to array broadside. The phase convention matches
    X = a(theta) s(t) + noise.
    """  # Docstring describes geometry and phase model.
    theta_rad = np.deg2rad(theta_deg)  # Convert degrees to radians.
    tau = positions * np.sin(theta_rad) / c  # Propagation delays for each sensor.
    return np.exp(-1j * 2 * np.pi * fc * tau)  # Complex steering vector.


def steering_matrix(
    scan_angles_deg: np.ndarray,  # Grid of scan angles in degrees.
    fc: float,  # Carrier/center frequency in Hz.
    c: float,  # Propagation speed in m/s.
    positions: np.ndarray,  # Sensor positions along the array axis (meters).
) -> np.ndarray:
    """Return steering vectors stacked column-wise for all scan angles."""  # Matrix builder.
    return np.column_stack(  # Stack each steering vector as a column.
        [steering_vector(theta, fc=fc, c=c, positions=positions) for theta in scan_angles_deg]
    )


def cbf_spectrum(
    X: np.ndarray,  # Snapshot data matrix (sensors x snapshots).
    scan_angles_deg: np.ndarray,  # Grid of scan angles in degrees.
    fc: float,  # Carrier/center frequency in Hz.
    c: float,  # Propagation speed in m/s.
    positions: np.ndarray,  # Sensor positions along the array axis (meters).
) -> np.ndarray:
    """Compute the conventional beamforming spatial spectrum.

    X must have shape (num_sensors, num_snapshots).
    """  # Docstring clarifies expected data shape.
    if X.ndim != 2:  # Enforce matrix input.
        raise ValueError("X must be a 2-D array with shape (num_sensors, num_snapshots).")
    if X.shape[0] != len(positions):  # Ensure sensors align with positions.
        raise ValueError("X.shape[0] must match the number of sensor positions.")

    Rxx = X @ X.conj().T / X.shape[1]  # Sample covariance matrix.
    A = steering_matrix(scan_angles_deg=scan_angles_deg, fc=fc, c=c, positions=positions)  # Steering matrix.
    RA = Rxx @ A  # Apply covariance to steering vectors.
    power = np.real(np.sum(A.conj() * RA, axis=0)) / (len(positions) ** 2)  # Beamformer power.
    return power  # Linear spectrum.


def estimate_doa_cbf(
    X: np.ndarray,  # Snapshot data matrix (sensors x snapshots).
    scan_angles_deg: np.ndarray,  # Grid of scan angles in degrees.
    fc: float,  # Carrier/center frequency in Hz.
    c: float,  # Propagation speed in m/s.
    positions: np.ndarray,  # Sensor positions along the array axis (meters).
) -> CBFResult:
    """Estimate DOA using conventional beamforming."""  # High-level estimator.
    power = cbf_spectrum(
        X=X,  # Data matrix.
        scan_angles_deg=scan_angles_deg,  # Scan grid.
        fc=fc,  # Carrier/center frequency.
        c=c,  # Propagation speed.
        positions=positions,  # Sensor positions.
    )
    theta_hat_deg = float(scan_angles_deg[np.argmax(power)])  # Peak location.
    power_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny) / np.max(power))  # Normalize to 0 dB peak.
    return CBFResult(
        theta_hat_deg=theta_hat_deg,  # Estimated DOA.
        scan_angles_deg=scan_angles_deg,  # Scan grid.
        power=power,  # Linear power.
        power_db=power_db,  # Normalized power in dB.
    )
