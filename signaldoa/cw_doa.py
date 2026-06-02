"""双信源 CW 窄带 DOA 仿真：简化版。

角度定义：相对均匀线阵 broadside 的角度，单位 deg。
"""
import numpy as np
try:
    from .cbf import cbf_spectrum, steering_vector
except ImportError:
    from cbf import cbf_spectrum, steering_vector


def add_awgn(X_clean: np.ndarray, snr_db: float, seed: int = None) -> np.ndarray:
    """给阵列接收矩阵添加复高斯白噪声。"""
    rng = np.random.default_rng(seed)

    signal_power = np.mean(np.abs(X_clean) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(X_clean.shape) + 1j * rng.standard_normal(X_clean.shape)
    )
    return X_clean + noise


def find_top_peaks(
    scan_angles: np.ndarray,
    power: np.ndarray,
    num_sources: int,
    min_separation_deg: float = 5.0,
) -> np.ndarray:
    """从空间谱中找出若干个相互分开的强峰。

    不直接取最大的 num_sources 个点，是因为同一个主瓣附近会有很多相邻高点。
    min_separation_deg 用来避免把同一个主瓣重复算成多个信源。
    """
    peak_indices = np.flatnonzero((power[1:-1] >= power[:-2]) & (power[1:-1] >= power[2:])) + 1

    if peak_indices.size == 0:
        peak_indices = np.array([np.argmax(power)])

    peak_indices = peak_indices[np.argsort(power[peak_indices])[::-1]]

    selected: list[int] = []
    for index in peak_indices:
        angle = scan_angles[index]
        separated = all(abs(angle - scan_angles[old]) >= min_separation_deg for old in selected)
        if separated:
            selected.append(int(index))
        if len(selected) == num_sources:
            break

    return np.sort(scan_angles[selected])


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

    theta_deg = np.array([15.0, 38.0])        # 两个真实 DOA
    amplitudes = np.array([1.0, 1.0])         # 两个信源幅度
    frequencies = np.array([fc, fc])

    fs = 240e3                  # 采样率，Hz
    duration = 0.020            # 观测时长，s
    snr_db = 10.0               # 信噪比，dB

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
    scan_angles = np.linspace(-90, 90, 1801)
    power = cbf_spectrum(
        X=X,
        scan_angles_deg=scan_angles,
        fc=fc,
        c=c,
        positions=positions,
    )
    power_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny) / np.max(power))

    theta_hat = find_top_peaks(
        scan_angles=scan_angles,
        power=power,
        num_sources=num_sources,
        min_separation_deg=5.0,
    )

    # -----------------------------
    # 5. 打印结果
    # -----------------------------
    print(f"fc = {fc / 1e3:.1f} kHz")
    print(f"wavelength = {wavelength:.4f} m")
    print(f"M = {M}, d = {d:.4f} m")
    print(f"X shape = {X.shape}  # (sensors, snapshots)")
    print(f"true DOAs = {theta_deg} deg")
    print(f"CBF estimated DOAs = {np.round(theta_hat, 2)} deg")
    print(f"absolute errors = {np.round(np.abs(theta_hat - theta_deg), 2)} deg")

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(scan_angles, power_db, linewidth=2, label="CBF spectrum")

        for i, theta in enumerate(theta_deg):
            plt.axvline(
                theta,
                color="tab:green",
                linestyle="--",
                linewidth=2,
                label="true DOA" if i == 0 else None,
            )

        for i, theta in enumerate(theta_hat):
            plt.axvline(
                theta,
                color="tab:red",
                linestyle=":",
                linewidth=2.5,
                label="estimated DOA" if i == 0 else None,
            )

        plt.ylim(-20, 3)
        plt.xlim(-90, 90)
        plt.xlabel("Scan angle (deg)")
        plt.ylabel("Normalized power (dB)")
        plt.title("Two-Source CW DOA Estimation with CBF")
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
        "theta_hat": theta_hat,
        "scan_angles": scan_angles,
        "power": power,
        "power_db": power_db,
        "fc": fc,
        "c": c,
        "wavelength": wavelength,
        "d": d,
        "snr_db": snr_db,
    }


if __name__ == "__main__":
    run_demo(plot=True)
