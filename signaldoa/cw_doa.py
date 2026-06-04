"""双信源 CW 窄带 DOA 仿真：CBF、MVDR、MUSIC 对比。
参数形状：
- X: (num_sensors, num_snapshots) # 阵列接收矩阵，多个信源叠加 + 噪声
- S: (num_sources, num_snapshots) # 信源时域信号矩阵
- scan_angles_deg: (num_scan_angles,)
- power: (num_scan_angles,)
- eigvals: (num_sensors,)  
- theta_hat: (num_estimated_sources,)  # 估计的 DOA 数量可能少于真实源数，取决于分辨率和 SNR。
- true_theta: (num_sources,)  # 真实 DOA 数量。
- positions: (num_sensors,)  # 阵元位置。
- time: (num_snapshots,)  # 快照时间点。

"""

from __future__ import annotations

import numpy as np

try:
    from .cbf import cbf_spectrum, steering_vector
    from .music import music_spectrum
    from .mvdr import mvdr_spectrum
    from .peak_search import find_top_peaks
except ImportError:
    from cbf import cbf_spectrum, steering_vector
    from music import music_spectrum
    from mvdr import mvdr_spectrum
    from peak_search import find_top_peaks


def add_awgn(X_clean: np.ndarray, snr_db: float, seed: int | None = None) -> np.ndarray:
    """给阵列接收矩阵添加复高斯白噪声。"""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(np.abs(X_clean) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(X_clean.shape) + 1j * rng.standard_normal(X_clean.shape)
    )
    return X_clean + noise


def run_demo(plot: bool = True) -> dict[str, np.ndarray | float]:
    """运行默认双信源 CW + CBF/MVDR/MUSIC 仿真。"""

    # 1. 直接设定仿真参数
    c = 1500.0                  # 声速，m/s
    fc = 30e3                   # 中心频率，Hz
    wavelength = c / fc         # 波长，m

    M = 16                      # 阵元数
    d = wavelength / 2          # 阵元间距，半波长避免栅瓣
    positions = np.arange(M) * d

    theta_deg = np.array([-10.0, 30.0, 40])   # 信源方位角，单位度
    scan_angles = np.linspace(-90, 90, 1801)  
    amplitudes = np.array([1.0, 1.0, 1.0])    # 信源幅度
    frequencies = np.array([fc, fc, fc])      # 窄带 CW 信号，频率相同

    fs = 240e3                  # 采样率，Hz
    duration = 0.020            # 观测时长，s
    snr_db = 20.0               # 信噪比，dB

    t = np.arange(0, duration, 1 / fs) # 时间向量，s
    num_sources = len(theta_deg)

    # 2. 生成彼此独立的窄带复包络 CW 信号，
    rng = np.random.default_rng()
    envelopes = (
        rng.standard_normal((num_sources, t.size))
        + 1j * rng.standard_normal((num_sources, t.size))
    ) / np.sqrt(2)
    phases = np.array([0.0, np.pi / 3, np.pi / 2])

    # 每个信源的时域信号矩阵 S， (num_sources, t.size)，满足 S_k(t) = A_k * e(t) * exp(j(2πf_k t + φ_k))
    S = amplitudes[:, None] * envelopes * np.exp(
        1j * (2 * np.pi * frequencies[:, None] * t[None, :] + phases[:, None])
    )

    # 3. 远场窄带阵列模型：X = sum_k a(theta_k) s_k(t) + W
    X_clean = np.zeros((M, t.size), dtype=complex)
    for k in range(num_sources):
        a = steering_vector(
            theta_deg=theta_deg[k],
            fc=frequencies[k],
            c=c,
            positions=positions,
        )
        X_clean += a[:, None] * S[k][None, :]

    X = add_awgn(X_clean, snr_db=snr_db, seed=None)

    # 4. CBF 空间谱
    cbf_power = cbf_spectrum(
        X=X,
        scan_angles_deg=scan_angles,
        fc=fc,
        c=c,
        positions=positions,
    )
    cbf_power_db = 10 * np.log10(np.maximum(cbf_power, np.finfo(float).tiny) / np.max(cbf_power))
    theta_hat_cbf = find_top_peaks(
        scan_angles=scan_angles,
        power=cbf_power,
        num_sources=num_sources,
    )

    # 5. MVDR 空间谱
    mvdr_power = mvdr_spectrum(
        X=X,
        scan_angles_deg=scan_angles,
        fc=fc,
        c=c,
        positions=positions,
        diagonal_loading=1e-2,
    )
    mvdr_power_db = 10 * np.log10(np.maximum(mvdr_power, np.finfo(float).tiny) / np.max(mvdr_power))
    theta_hat_mvdr = find_top_peaks(
        scan_angles=scan_angles,
        power=mvdr_power,
        num_sources=num_sources,
    )

    # 6. MUSIC 空间谱
    music_power, music_eigvals = music_spectrum(
        X=X,
        scan_angles_deg=scan_angles,
        fc=fc,
        c=c,
        positions=positions,
        num_sources=num_sources,
    )
    music_power_db = 10 * np.log10(np.maximum(music_power, np.finfo(float).tiny) / np.max(music_power))
    theta_hat_music = find_top_peaks(
        scan_angles=scan_angles,
        power=music_power,
        num_sources=num_sources,
    )

    # 7. 打印结果
    print(f"fc = {fc / 1e3:.1f} kHz")
    print(f"wavelength = {wavelength:.4f} m")
    print(f"M = {M}, d = {d:.4f} m")
    print(f"X shape = {X.shape}  # (sensors, snapshots)")
    print(f"true DOAs = {theta_deg} deg")
    print(f"CBF estimated DOAs = {np.round(theta_hat_cbf, 2)} deg")
    if len(theta_hat_cbf) == len(theta_deg):
        print(f"CBF absolute errors = {np.round(np.abs(theta_hat_cbf - theta_deg), 2)} deg")
    else:
        print("CBF absolute errors = not available because not all sources were resolved")

    print(f"MVDR estimated DOAs = {np.round(theta_hat_mvdr, 2)} deg")
    if len(theta_hat_mvdr) == len(theta_deg):
        print(f"MVDR absolute errors = {np.round(np.abs(theta_hat_mvdr - theta_deg), 2)} deg")
    else:
        print("MVDR absolute errors = not available because not all sources were resolved")

    print(f"MUSIC estimated DOAs = {np.round(theta_hat_music, 2)} deg")
    if len(theta_hat_music) == len(theta_deg):
        print(f"MUSIC absolute errors = {np.round(np.abs(theta_hat_music - theta_deg), 2)} deg")
    else:
        print("MUSIC absolute errors = not available because not all sources were resolved")
    print("Largest covariance eigenvalues =", np.round(music_eigvals[:4], 4))

    if plot:
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = ["DejaVu Sans", "SimSun"]
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimSun"]
        plt.rcParams["axes.unicode_minus"] = False

        plt.figure(figsize=(10, 5))
        plt.plot(scan_angles, cbf_power_db, linewidth=1.5, label="CBF 空间谱")
        plt.plot(scan_angles, mvdr_power_db, linewidth=1.5, label="MVDR 空间谱")
        plt.plot(scan_angles, music_power_db, linewidth=1.5, label="MUSIC 空间谱")

        all_power_db = np.concatenate([cbf_power_db, mvdr_power_db, music_power_db])
        finite_power_db = all_power_db[np.isfinite(all_power_db)]
        y_min = max(np.percentile(finite_power_db, 1) - 1, -80)
        y_max = np.max(finite_power_db) + 0.1

        for i, theta in enumerate(theta_deg):
            plt.axvline(
                theta,
                color="tab:red",
                linestyle="--",
                linewidth=0.8,
                label="真实方位" if i == 0 else None,
            )
            plt.text(
                theta,
                y_max,
                "*",
                color="tab:red",
                fontsize=18,
                ha="center",
                va="top",
            )

        plt.ylim(y_min, y_max)
        plt.xlim(scan_angles[0], scan_angles[-1])
        plt.xlabel("扫描角度 (deg)")
        plt.ylabel("归一化功率 (dB)")
        plt.title("双信源 CW DOA 估计：CBF、MVDR 与 MUSIC 对比")
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    return {
        "X": X,
        "X_clean": X_clean,
        "source_signals": S,
        "time": t,
        "positions": positions,
        "theta_deg": theta_deg,
        "theta_hat_cbf": theta_hat_cbf,
        "theta_hat_mvdr": theta_hat_mvdr,
        "theta_hat_music": theta_hat_music,
        "scan_angles": scan_angles,
        "cbf_power": cbf_power,
        "cbf_power_db": cbf_power_db,
        "mvdr_power": mvdr_power,
        "mvdr_power_db": mvdr_power_db,
        "music_power": music_power,
        "music_power_db": music_power_db,
        "music_eigvals": music_eigvals,
        "fc": fc,
        "c": c,
        "wavelength": wavelength,
        "d": d,
        "snr_db": snr_db,
    }


if __name__ == "__main__":
    run_demo(plot=True)
