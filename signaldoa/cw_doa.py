"""双信源 CW 窄带 DOA 仿真：简化版。

角度定义：相对均匀线阵 broadside 的角度，单位 deg。
"""
import numpy as np
try:
    from .cbf import cbf_spectrum, steering_vector
    from .mvdr import mvdr_spectrum
    from .peak_search import find_top_peaks
except ImportError:
    from cbf import cbf_spectrum, steering_vector
    from mvdr import mvdr_spectrum
    from peak_search import find_top_peaks


def add_awgn(X_clean: np.ndarray, snr_db: float, seed: int = None) -> np.ndarray:
    """给阵列接收矩阵添加复高斯白噪声。"""
    rng = np.random.default_rng(seed)

    signal_power = np.mean(np.abs(X_clean) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(X_clean.shape) + 1j * rng.standard_normal(X_clean.shape)
    )
    return X_clean + noise


def run_demo(plot: bool = True) -> dict[str, np.ndarray | float]:
    """运行默认双信源 CW + CBF 仿真。"""

    # -----------------------------
    # 1. 直接设定仿真参数
    # -----------------------------
    c = 1500.0                  # 声速，m/s
    fc = 30e3                   # 中心频率，Hz
    wavelength = c / fc         # 波长，m

    M = 16                      # 阵元数
    d = wavelength / 2          # 阵元间距，半波长避免栅瓣
    positions = np.arange(M) * d

    theta_deg = np.array([30.0, 50])        # 两个真实 DOA
    scan_angles = np.linspace(-90, 90, 1801)
    amplitudes = np.array([1.0, 1.0])         # 两个信源幅度
    frequencies = np.array([fc, fc])

    fs = 240e3                  # 采样率，Hz
    duration = 0.020            # 观测时长，s
    snr_db = 0.0               # 信噪比，dB

    t = np.arange(0, duration, 1 / fs)
    num_sources = len(theta_deg)

    # -----------------------------
    # 2. 生成多个 CW 信源
    # -----------------------------
    # S 的形状是 (信源数, 快拍数)。
    # 第 k 行是第 k 个信源的时间序列。
    phases = np.array([0.0, np.pi / 3])
    S = amplitudes[:, None] * np.exp(
        1j * (2 * np.pi * frequencies[:, None] * t[None, :] + phases[:, None])
    )

    # -----------------------------
    # 3. 按远场窄带阵列模型生成接收数据
    # -----------------------------
    # 单个信源模型：
    # X_k = a(theta_k) s_k(t)
    #
    # 多个信源叠加：
    # X_clean = sum_k a(theta_k) s_k(t)
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

    # -----------------------------
    # 4. CBF 扫描空间谱
    # -----------------------------
    # 181 个扫描点对应 1 deg 步长；1801 个扫描点则对应 0.1 deg 步长。
    # 扫描更细只会让谱峰读数更细，不会真正提高阵列本身的角度分辨率。
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

    # -----------------------------
    # 5. 打印结果
    # -----------------------------
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

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(scan_angles, cbf_power_db, linewidth=2, label="CBF spectrum")
        plt.plot(scan_angles, mvdr_power_db, linewidth=2, label="MVDR spectrum")

        for i, theta in enumerate(theta_deg):
            plt.axvline(
                theta,
                color="tab:green",
                linestyle="--",
                linewidth=2,
                label="true DOA" if i == 0 else None,
            )

        for i, theta in enumerate(theta_hat_cbf):
            plt.axvline(
                theta,
                color="tab:red",
                linestyle=":",
                linewidth=2.5,
                label="CBF estimated DOA" if i == 0 else None,
            )

        for i, theta in enumerate(theta_hat_mvdr):
            plt.axvline(
                theta,
                color="tab:blue",
                linestyle="-.",
                linewidth=2,
                label="MVDR estimated DOA" if i == 0 else None,
            )

        all_power_db = np.concatenate([cbf_power_db, mvdr_power_db])
        finite_power_db = all_power_db[np.isfinite(all_power_db)]
        y_min = max(np.percentile(finite_power_db, 1) - 3, -80)
        y_max = np.max(finite_power_db) + 3

        plt.ylim(y_min, y_max)
        plt.xlim(scan_angles[0], scan_angles[-1])
        plt.xlabel("Scan angle (deg)")
        plt.ylabel("Normalized power (dB)")
        plt.title("Two-Source CW DOA Estimation: CBF vs MVDR")
        plt.grid(True)
        plt.legend(loc="upper right")
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
        "scan_angles": scan_angles,
        "cbf_power": cbf_power,
        "cbf_power_db": cbf_power_db,
        "mvdr_power": mvdr_power,
        "mvdr_power_db": mvdr_power_db,
        "fc": fc,
        "c": c,
        "wavelength": wavelength,
        "d": d,
        "snr_db": snr_db,
    }


if __name__ == "__main__":
    run_demo(plot=True)
