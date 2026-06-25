import json
import re
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "experiments" / "LFM.ipynb"


def decode_unicode_escapes(text: str) -> str:
    """Decode only \\uXXXX escapes, leaving LaTeX backslashes untouched."""
    return re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        text,
    )


def source_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.strip().split("\n")]


def md(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source_lines(decode_unicode_escapes(text)),
    }


def code(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


frft_cells = [
    md(
        "frft-01-overview",
        r"""
# FRFT \u4e0e LFM DOA\uff1a\u4ece\u805a\u7126\u5230\u65b9\u4f4d\u4f30\u8ba1

\u8fd9\u4e00\u90e8\u5206\u91cd\u65b0\u6309\u4e00\u6761\u4e3b\u7ebf\u7ec4\u7ec7\uff1a

1. \u5148\u770b FRFT \u4e3a\u4ec0\u4e48\u80fd\u805a\u7126 LFM\u3002
2. \u518d\u7528\u9636\u6b21\u626b\u63cf\u627e\u6700\u9002\u5408\u5f53\u524d LFM \u7684 FRFT \u9636\u6b21\u3002
3. \u7136\u540e\u5b9e\u73b0\u4e24\u79cd FRFT-CBF\uff1a
   - **\u5bbd\u5e26\u5ef6\u65f6\u6c42\u548c + FRFT**\uff1a\u5148\u6309\u5019\u9009\u89d2\u5bf9\u9f50\u9635\u5143\u4fe1\u53f7\uff0c\u518d\u505a FRFT \u53d6\u805a\u7126\u80fd\u91cf\u3002
   - **\u53c2\u8003\u6587\u732e\u4e2d\u7684 FRF \u57df CBF**\uff1a\u6bcf\u4e2a\u9635\u5143\u5148\u505a FRFT\uff0c\u5728\u805a\u7126 bin \u5904\u6784\u9020\u9635\u5217\u5feb\u7167\uff0c\u518d\u7528 $a^H R a$ \u626b\u63cf\u7a7a\u95f4\u8c31\u3002

\u8fd9\u91cc\u7684\u4ee3\u7801\u4f18\u5148\u4fdd\u6301\u6982\u5ff5\u6e05\u695a\u548c\u53ef\u9a8c\u8bc1\u3002FRFT \u91c7\u7528\u76f4\u63a5\u77e9\u9635\u5f62\u5f0f\uff0c\u9002\u5408\u6559\u5b66\u4e0e\u5c0f\u89c4\u6a21\u4eff\u771f\uff1b\u5982\u679c\u540e\u7eed\u505a\u5927\u89c4\u6a21\u5b9e\u9a8c\uff0c\u518d\u6362\u6210\u5feb\u901f FRFT \u5b9e\u73b0\u3002
""",
    ),
    md(
        "frft-02-theory",
        r"""
## 1. FRFT \u4e3a\u4ec0\u4e48\u80fd\u805a\u7126 LFM

LFM \u4fe1\u53f7\u7684\u590d\u89e3\u6790\u5f62\u5f0f\u662f\uff1a

$$
s(t)=A\exp\left[j2\pi\left(f_0t+\frac{1}{2}\mu t^2\right)+j\phi\right]
$$

\u5b83\u7684\u77ac\u65f6\u9891\u7387\u4e3a\uff1a

$$
f(t)=f_0+\mu t
$$

\u6240\u4ee5 LFM \u5728\u65f6\u9891\u5e73\u9762\u91cc\u662f\u4e00\u6761\u659c\u7ebf\u3002\u666e\u901a Fourier \u53d8\u6362\u76f8\u5f53\u4e8e\u628a\u4fe1\u53f7\u6295\u5f71\u5230\u9891\u7387\u8f74\uff1bFRFT \u53ef\u4ee5\u7406\u89e3\u4e3a\u628a\u65f6\u9891\u5750\u6807\u8f74\u65cb\u8f6c\u4e00\u4e2a\u89d2\u5ea6\u540e\u518d\u89c2\u5bdf\u4fe1\u53f7\u3002

FRFT \u9636\u6b21 $p$ \u4e0e\u65cb\u8f6c\u89d2\u7684\u5173\u7cfb\u662f\uff1a

$$
\alpha=p\frac{\pi}{2}
$$

\u5176\u6838\u51fd\u6570\u53ef\u5199\u6210\uff1a

$$
K_\alpha(t,u)=A_\alpha
\exp\left[
j\pi\left(t^2\cot\alpha-2tu\csc\alpha+u^2\cot\alpha\right)
\right]
$$

\u5b83\u540c\u6837\u542b\u6709\u4e8c\u6b21\u76f8\u4f4d\u9879\u3002\u5f53 FRFT \u9636\u6b21\u4e0e LFM \u8c03\u9891\u659c\u7387\u5339\u914d\u65f6\uff0c\u539f\u672c\u5206\u6563\u5728\u4e00\u6761\u659c\u7ebf\u4e0a\u7684 LFM \u80fd\u91cf\u4f1a\u5728\u5206\u6570\u9636\u57df\u91cc\u88ab\u538b\u7f29\u5230\u5c11\u6570 bin \u9644\u8fd1\u3002

\u7b80\u5355\u8bf4\uff1a**FFT \u662f\u627e\u56fa\u5b9a\u9891\u7387\u7684\u80fd\u91cf\uff0cFRFT \u662f\u627e\u67d0\u4e2a chirp \u659c\u7387\u4e0a\u7684\u80fd\u91cf\u3002**
""",
    ),
    code(
        "frft-03-implementation",
        r'''
# -----------------------------
# FRFT helper functions
# -----------------------------
def dft_unitary(x):
    """Unitary DFT with centered frequency bins."""
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x))) / np.sqrt(len(x))


def idft_unitary(x):
    """Unitary inverse DFT with centered bins."""
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x))) * np.sqrt(len(x))


def frft_matrix_direct(N_local, p):
    """Direct DFRFT matrix used by this notebook.

    The sample coordinates are dimensionless and centered. This convention is
    pedagogical; all later steering/manifold calculations are built with the
    same matrix so the phase convention stays self-consistent.
    """
    p_mod = np.mod(p, 4.0)
    if np.isclose(p_mod, 0.0):
        return np.eye(N_local, dtype=complex)
    if np.isclose(p_mod, 1.0):
        eye = np.eye(N_local, dtype=complex)
        return np.column_stack([dft_unitary(eye[:, k]) for k in range(N_local)])
    if np.isclose(p_mod, 2.0):
        return np.eye(N_local, dtype=complex)[::-1]
    if np.isclose(p_mod, 3.0):
        eye = np.eye(N_local, dtype=complex)
        return np.column_stack([idft_unitary(eye[:, k]) for k in range(N_local)])

    alpha = p_mod * np.pi / 2
    sin_alpha = np.sin(alpha)
    cot_alpha = 1 / np.tan(alpha)
    csc_alpha = 1 / sin_alpha

    n = np.arange(N_local) - (N_local - 1) / 2
    t_norm = n / np.sqrt(N_local)
    u_norm = n / np.sqrt(N_local)
    T, U = np.meshgrid(t_norm, u_norm, indexing="ij")
    phase = np.pi * (T**2 * cot_alpha - 2 * T * U * csc_alpha + U**2 * cot_alpha)
    kernel = np.exp(1j * phase)
    return kernel.conj().T / np.sqrt(N_local * abs(sin_alpha))


def frft_direct(x, p):
    """Apply the direct DFRFT matrix."""
    x = np.asarray(x, dtype=complex)
    return frft_matrix_direct(x.size, p) @ x


def normalized_db(power_like):
    power_like = np.asarray(power_like)
    return 10 * np.log10(np.maximum(power_like, np.finfo(float).tiny) / np.max(power_like))


def parabolic_peak_interpolation(x_grid, y):
    """Sub-grid peak interpolation using three neighboring linear-power points."""
    idx = int(np.argmax(y))
    if idx == 0 or idx == len(y) - 1:
        return float(x_grid[idx])
    y_left, y_mid, y_right = y[idx - 1], y[idx], y[idx + 1]
    denom = y_left - 2 * y_mid + y_right
    if abs(denom) < np.finfo(float).eps:
        return float(x_grid[idx])
    step = float(x_grid[1] - x_grid[0])
    offset = 0.5 * (y_left - y_right) / denom
    offset = float(np.clip(offset, -1.0, 1.0))
    return float(x_grid[idx] + offset * step)


# Use a shorter segment for O(N^2) FRFT demonstrations.
frft_stride = 4
s_frft = s[::frft_stride]
t_frft = t[::frft_stride]
fs_frft = fs / frft_stride
N_frft = s_frft.size
s_frft_unit = s_frft / np.sqrt(np.sum(np.abs(s_frft) ** 2))

print("FRFT analysis segment")
print(f"N_frft = {N_frft}, fs_frft = {fs_frft / 1e3:.1f} kHz, stride = {frft_stride}")
print(f"p=0 check error = {np.linalg.norm(frft_direct(s_frft_unit, 0) - s_frft_unit):.2e}")
print(f"p=2 check error = {np.linalg.norm(frft_direct(s_frft_unit, 2) - s_frft_unit[::-1]):.2e}")
''',
    ),
    code(
        "frft-04-order-scan",
        r'''
# -----------------------------
# Find the FRFT order that best focuses the LFM
# -----------------------------
orders = np.linspace(0.05, 1.95, 121)
peak_to_mean = np.zeros_like(orders)
entropy = np.zeros_like(orders)

for i, p in enumerate(orders):
    Xp = frft_direct(s_frft_unit, p)
    power = np.abs(Xp) ** 2
    power = power / np.sum(power)
    peak_to_mean[i] = np.max(power) / np.mean(power)
    entropy[i] = -np.sum(power * np.log(power + np.finfo(float).tiny))

best_idx = int(np.argmax(peak_to_mean))
p_best = float(orders[best_idx])
alpha_best = p_best * np.pi / 2

print(f"Best FRFT order = {p_best:.3f}")
print(f"Best rotation angle alpha = {np.rad2deg(alpha_best):.2f} deg")
print(f"Peak-to-mean = {peak_to_mean[best_idx]:.2f}")
print(f"Entropy = {entropy[best_idx]:.2f}")

plt.figure(figsize=(10, 4))
plt.plot(orders, peak_to_mean, linewidth=1.8)
plt.axvline(p_best, color="tab:red", linestyle="--", linewidth=1.2, label=f"best p = {p_best:.3f}")
plt.xlabel("FRFT order p")
plt.ylabel("Peak-to-mean concentration")
plt.title("FRFT order scan for LFM")
plt.legend(loc="best")
plt.tight_layout()
''',
    ),
    code(
        "frft-05-focus-plot",
        r'''
# -----------------------------
# Compare ordinary FFT with best-order FRFT
# -----------------------------
X_fft = frft_direct(s_frft_unit, 1.0)
X_best = frft_direct(s_frft_unit, p_best)

freq_fft = np.fft.fftshift(np.fft.fftfreq(N_frft, d=1 / fs_frft)) / 1e3
u_axis = np.arange(N_frft) - (N_frft - 1) / 2

fft_mag_db = 20 * np.log10(np.maximum(np.abs(X_fft), np.finfo(float).tiny) / np.max(np.abs(X_fft)))
best_mag_db = 20 * np.log10(np.maximum(np.abs(X_best), np.finfo(float).tiny) / np.max(np.abs(X_best)))

plt.figure(figsize=(11, 5))

plt.subplot(1, 2, 1)
plt.plot(freq_fft, fft_mag_db, linewidth=1.4)
plt.axvline(f0 / 1e3, color="tab:red", linestyle="--", linewidth=0.8, label="start f")
plt.axvline((f0 + mu * duration) / 1e3, color="tab:green", linestyle="--", linewidth=0.8, label="end f")
plt.xlim(0, 60)
plt.ylim(-60, 3)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Normalized magnitude (dB)")
plt.title("FFT: energy spread over sweep band")
plt.legend(loc="upper right")

plt.subplot(1, 2, 2)
plt.plot(u_axis, best_mag_db, linewidth=1.4, color="tab:purple")
plt.ylim(-60, 3)
plt.xlabel("FRFT-domain bin")
plt.ylabel("Normalized magnitude (dB)")
plt.title(f"FRFT: focused energy, p = {p_best:.3f}")
plt.tight_layout()
''',
    ),
    md(
        "frft-06-cbf-theory",
        r"""
## 2. \u4e24\u79cd FRFT-CBF \u601d\u8def

### A. \u5bbd\u5e26\u5ef6\u65f6\u6c42\u548c + FRFT

\u8fd9\u4e2a\u7248\u672c\u66f4\u76f4\u89c2\uff1a\u5bf9\u6bcf\u4e2a\u626b\u63cf\u89d2 $\theta$ \u5148\u505a\u5bbd\u5e26 CBF \u5bf9\u9f50\uff1a

$$
y_\theta(t)=\frac{1}{M}\sum_{m=0}^{M-1}x_m(t+\tau_m(\theta))
$$

ULA \u7684\u5019\u9009\u65f6\u5ef6\u4e3a\uff1a

$$
\tau_m(\theta)=\frac{p_m\sin\theta}{c}
$$

\u7136\u540e\u5bf9 $y_\theta(t)$ \u505a FRFT\u3002\u4e3a\u4e86\u66f4\u50cf\u201c\u805a\u7126\u540e\u7684\u7a7a\u95f4\u8c31\u201d\uff0c\u4e0d\u8ba9\u6bcf\u4e2a\u89d2\u5ea6\u968f\u4fbf\u627e\u81ea\u5df1\u7684\u6700\u5927 bin\uff0c\u6211\u4eec\u56fa\u5b9a\u5168\u5c40\u805a\u7126 bin $u_0$\uff1a

$$
P_{delay\text{-}sum}(\theta)
=\left|\mathcal F^{p_{best}}\{y_\theta(t)\}(u_0)\right|^2
$$

### B. \u53c2\u8003\u6587\u732e\u4e2d\u7684 FRF \u57df CBF

\u53c2\u8003\u6587\u732e\u7684\u5199\u6cd5\u662f\uff1a\u6bcf\u4e2a\u9635\u5143\u5148\u505a FRFT\uff0c\u5f97\u5230\u5206\u6570\u9636\u57df\u9635\u5217\u5feb\u7167\uff1a

$$
\mathbf X(u,\alpha)=
\begin{bmatrix}
X_0(u,\alpha) & X_1(u,\alpha) & \cdots & X_{M-1}(u,\alpha)
\end{bmatrix}^T
$$

\u7136\u540e\u4eff\u7167\u7a84\u5e26 CBF\uff1a

$$
y(u,\alpha,\theta)=\mathbf a^H(u,\alpha,\theta)\mathbf X(u,\alpha)
$$

$$
P_{FRF\text{-}CBF}(\theta)
=\mathbf a^H(u,\alpha,\theta)\mathbf R(u,\alpha)\mathbf a(u,\alpha,\theta)
$$

\u5176\u4e2d\uff1a

$$
\mathbf R(u,\alpha)=E[\mathbf X(u,\alpha)\mathbf X^H(u,\alpha)]
$$

\u6837\u672c\u6709\u9650\u65f6\u53ef\u4ee5\u5728\u805a\u7126\u5cf0\u9644\u8fd1\u53d6\u51e0\u4e2a bin \u4f30\u8ba1\uff1a

$$
\hat{\mathbf R}
=\frac{1}{L}\sum_{\ell=1}^{L}
\mathbf X(u_\ell,\alpha)\mathbf X^H(u_\ell,\alpha)
$$

\u8981\u6ce8\u610f\uff1aFRFT \u57df\u7684\u5ef6\u65f6\u54cd\u5e94\u4e0d\u662f\u7b80\u5355\u7684 $e^{-j2\pi f_c\tau}$\u3002\u5b83\u4f1a\u6d89\u53ca\u5206\u6570\u9636\u5750\u6807\u504f\u79fb\u548c\u4e8c\u6b21\u76f8\u4f4d\u3002\u6240\u4ee5\u4ee3\u7801\u91cc\u4e0d\u76f4\u63a5\u5957\u666e\u901a\u7a84\u5e26 steering vector\uff0c\u800c\u662f\u7528\u4e0e\u5f53\u524d DFRFT \u5b9e\u73b0\u4e00\u81f4\u7684**\u6821\u51c6\u9635\u5217\u6d41\u5f62**\u6765\u6784\u9020 $\mathbf a(u,\alpha,\theta)$\u3002
""",
    ),
    code(
        "frft-07-array-simulation",
        r'''
# -----------------------------
# ULA LFM array simulation
# -----------------------------
c_lfm = 1500.0
M_lfm = 16
d_lfm = 0.025
positions_lfm = np.arange(M_lfm) * d_lfm

theta_true_lfm_deg = 25.0
snr_lfm_db = 20.0


def analytic_lfm_at(t_eval):
    """Evaluate the finite-duration analytic LFM at arbitrary times."""
    t_eval = np.asarray(t_eval)
    y = np.zeros(t_eval.shape, dtype=complex)
    inside = (t_eval >= 0.0) & (t_eval < duration)
    phase_eval = 2 * np.pi * (f0 * t_eval[inside] + 0.5 * mu * t_eval[inside] ** 2) + phi
    y[inside] = A * np.exp(1j * phase_eval)
    return y


def add_complex_awgn_matrix(X_clean, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    signal_power = np.mean(np.abs(X_clean) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(X_clean.shape) + 1j * rng.standard_normal(X_clean.shape)
    )
    return X_clean + noise


def interp_complex_1d(t_axis, x, t_query):
    """Linear interpolation for complex-valued sampled data."""
    real = np.interp(t_query, t_axis, np.real(x), left=0.0, right=0.0)
    imag = np.interp(t_query, t_axis, np.imag(x), left=0.0, right=0.0)
    return real + 1j * imag


theta_true_rad = np.deg2rad(theta_true_lfm_deg)
tau_true = positions_lfm * np.sin(theta_true_rad) / c_lfm
X_lfm_clean = np.vstack([analytic_lfm_at(t - tau_m) for tau_m in tau_true])
X_lfm = add_complex_awgn_matrix(X_lfm_clean, snr_db=snr_lfm_db, seed=2026)

scan_angles_frft_cbf = np.arange(-60.0, 60.0 + 0.05, 0.1)
max_scan_delay = np.max(np.abs(positions_lfm)) * np.sin(np.deg2rad(np.max(np.abs(scan_angles_frft_cbf)))) / c_lfm
t_cbf_all = t[::frft_stride]
valid_cbf = (t_cbf_all >= max_scan_delay) & (t_cbf_all <= duration - max_scan_delay)
t_cbf = t_cbf_all[valid_cbf]
N_cbf = t_cbf.size

# Re-estimate best FRFT order on the exact CBF time window.
s_cbf_ref = analytic_lfm_at(t_cbf)
s_cbf_ref = s_cbf_ref / np.sqrt(np.sum(np.abs(s_cbf_ref) ** 2))
cbf_order_metric = np.zeros_like(orders)
for i_order, p in enumerate(orders):
    Xp_ref = frft_direct(s_cbf_ref, p)
    power_ref = np.abs(Xp_ref) ** 2
    power_ref = power_ref / np.sum(power_ref)
    cbf_order_metric[i_order] = np.max(power_ref) / np.mean(power_ref)

p_cbf_best = float(orders[int(np.argmax(cbf_order_metric))])
F_cbf_best = frft_matrix_direct(N_cbf, p_cbf_best)

print("FRFT-CBF array simulation")
print(f"M = {M_lfm}, d = {d_lfm:.4f} m")
print(f"true DOA = {theta_true_lfm_deg:.1f} deg, SNR = {snr_lfm_db:.1f} dB")
print(f"CBF valid samples = {N_cbf}, guard = {max_scan_delay * 1e6:.2f} us")
print(f"full-pulse p_best = {p_best:.3f}, CBF-window p_best = {p_cbf_best:.3f}")
''',
    ),
    code(
        "frft-08-delay-sum-cbf",
        r'''
# -----------------------------
# Method A: delay-sum first, then FRFT focused-bin power
# -----------------------------
delay_sum_frft_matrix = np.zeros((scan_angles_frft_cbf.size, N_cbf))

for i, theta_deg in enumerate(scan_angles_frft_cbf):
    theta_rad = np.deg2rad(theta_deg)
    tau_scan = positions_lfm * np.sin(theta_rad) / c_lfm

    aligned_sum = np.zeros(N_cbf, dtype=complex)
    for m in range(M_lfm):
        aligned_sum += interp_complex_1d(t, X_lfm[m], t_cbf + tau_scan[m])

    beam_output = aligned_sum / M_lfm
    beam_frft = F_cbf_best @ beam_output
    delay_sum_frft_matrix[i] = np.abs(beam_frft) ** 2

# Diagnostic spectrum: each angle chooses its strongest FRFT bin.
delay_sum_maxbin_power = np.max(delay_sum_frft_matrix, axis=1)

# Focused spectrum: all angles are compared at one global focused bin.
global_angle_index, global_focus_bin = np.unravel_index(
    np.argmax(delay_sum_frft_matrix),
    delay_sum_frft_matrix.shape,
)
delay_sum_focused_power = delay_sum_frft_matrix[:, global_focus_bin]

theta_hat_delay_sum_grid = float(scan_angles_frft_cbf[np.argmax(delay_sum_focused_power)])
theta_hat_delay_sum = parabolic_peak_interpolation(scan_angles_frft_cbf, delay_sum_focused_power)

print("Method A: delay-sum + FRFT")
print(f"global focused bin = {global_focus_bin}")
print(f"grid estimate = {theta_hat_delay_sum_grid:.2f} deg")
print(f"interpolated estimate = {theta_hat_delay_sum:.2f} deg")
print(f"absolute error = {abs(theta_hat_delay_sum - theta_true_lfm_deg):.2f} deg")

plt.figure(figsize=(10, 4.5))
plt.plot(
    scan_angles_frft_cbf,
    normalized_db(delay_sum_maxbin_power),
    color="tab:blue",
    linestyle="--",
    linewidth=1.2,
    alpha=0.75,
    label="max over FRFT bins",
)
plt.plot(
    scan_angles_frft_cbf,
    normalized_db(delay_sum_focused_power),
    color="tab:red",
    linewidth=1.8,
    label="fixed focused bin",
)
plt.axvline(theta_true_lfm_deg, color="tab:green", linestyle="--", linewidth=1.2, label=f"true {theta_true_lfm_deg:.1f} deg")
plt.axvline(theta_hat_delay_sum, color="k", linestyle=":", linewidth=1.8, label=f"estimated {theta_hat_delay_sum:.2f} deg")
plt.ylim(-40, 3)
plt.xlim(scan_angles_frft_cbf[0], scan_angles_frft_cbf[-1])
plt.xlabel("Scan angle (deg)")
plt.ylabel("Normalized power (dB)")
plt.title("Method A: delay-sum then FRFT")
plt.legend(loc="best")
plt.tight_layout()
''',
    ),
    code(
        "frft-09-frf-domain-cbf",
        r'''
# -----------------------------
# Method B: paper-style FRF-domain CBF, P(theta) = a^H R a
# -----------------------------
# 1. Transform each sensor into the FRFT domain on the same valid time window.
X_frf_time = np.vstack([
    interp_complex_1d(t, X_lfm[m], t_cbf)
    for m in range(M_lfm)
])
X_frf = (F_cbf_best @ X_frf_time.T).T  # shape: (M, N_cbf)

# 2. Use a small bin group around the focused FRFT peak to estimate covariance.
frf_focus_center = int(np.argmax(np.sum(np.abs(X_frf) ** 2, axis=0)))
frf_focus_half_width = 1
frf_focus_bins = np.arange(
    max(0, frf_focus_center - frf_focus_half_width),
    min(N_cbf, frf_focus_center + frf_focus_half_width + 1),
)
F_focus = F_cbf_best[frf_focus_bins, :]
X_focus = X_frf[:, frf_focus_bins]  # shape: (M, L)


def calibrated_frf_manifold(theta_deg):
    """FRF-domain steering manifold matched to this notebook's DFRFT convention."""
    theta_rad = np.deg2rad(theta_deg)
    tau_candidate = positions_lfm * np.sin(theta_rad) / c_lfm
    A_time = np.vstack([
        analytic_lfm_at(t_cbf - tau_m)
        for tau_m in tau_candidate
    ])
    A_focus = (F_focus @ A_time.T).T  # shape: (M, L)
    norms = np.linalg.norm(A_focus, axis=0, keepdims=True)
    return A_focus / np.maximum(norms, np.finfo(float).eps)


frf_cbf_power = np.zeros(scan_angles_frft_cbf.size)

for i, theta_deg in enumerate(scan_angles_frft_cbf):
    A_focus = calibrated_frf_manifold(theta_deg)
    power_accum = 0.0
    for k in range(frf_focus_bins.size):
        a_k = A_focus[:, k]
        x_k = X_focus[:, k]
        # a^H (x x^H) a = |a^H x|^2
        power_accum += np.abs(np.vdot(a_k, x_k)) ** 2
    frf_cbf_power[i] = power_accum / frf_focus_bins.size

theta_hat_frf_grid = float(scan_angles_frft_cbf[np.argmax(frf_cbf_power)])
theta_hat_frf = parabolic_peak_interpolation(scan_angles_frft_cbf, frf_cbf_power)

print("Method B: paper-style FRF-domain CBF")
print(f"focused bin center = {frf_focus_center}, bins = {frf_focus_bins.tolist()}")
print(f"grid estimate = {theta_hat_frf_grid:.2f} deg")
print(f"interpolated estimate = {theta_hat_frf:.2f} deg")
print(f"absolute error = {abs(theta_hat_frf - theta_true_lfm_deg):.2f} deg")

plt.figure(figsize=(10, 4.5))
plt.plot(
    scan_angles_frft_cbf,
    normalized_db(delay_sum_focused_power),
    color="tab:red",
    linewidth=1.4,
    alpha=0.85,
    label="A: delay-sum + FRFT",
)
plt.plot(
    scan_angles_frft_cbf,
    normalized_db(frf_cbf_power),
    color="tab:purple",
    linewidth=1.8,
    label="B: FRF-domain CBF",
)
plt.axvline(theta_true_lfm_deg, color="tab:green", linestyle="--", linewidth=1.2, label=f"true {theta_true_lfm_deg:.1f} deg")
plt.axvline(theta_hat_frf, color="k", linestyle=":", linewidth=1.8, label=f"FRF-CBF estimated {theta_hat_frf:.2f} deg")
plt.ylim(-40, 3)
plt.xlim(scan_angles_frft_cbf[0], scan_angles_frft_cbf[-1])
plt.xlabel("Scan angle (deg)")
plt.ylabel("Normalized spatial spectrum (dB)")
plt.title("FRFT-CBF methods for LFM DOA")
plt.legend(loc="best")
plt.tight_layout()
''',
    ),
    md(
        "frft-10-summary",
        r"""
## 3. \u5c0f\u7ed3

\u4e24\u79cd\u65b9\u6cd5\u7684\u5dee\u522b\u53ef\u4ee5\u8fd9\u6837\u8bb0\uff1a

- **\u5ef6\u65f6\u6c42\u548c + FRFT**\uff1a\u5bf9\u6bcf\u4e2a\u89d2\u5ea6\u5148\u628a\u9635\u5143\u4fe1\u53f7\u5728\u65f6\u57df\u5bf9\u9f50\uff0c\u518d\u770b\u5bf9\u9f50\u540e\u7684 LFM \u5728 FRFT \u805a\u7126 bin \u4e0a\u6709\u591a\u5f3a\u3002
- **FRF \u57df CBF**\uff1a\u6bcf\u4e2a\u9635\u5143\u5148\u505a FRFT\uff0c\u7136\u540e\u628a\u805a\u7126 bin \u5904\u7684\u9635\u5217\u6570\u636e\u5f53\u6210\u201c\u5206\u6570\u9636\u57df\u7a84\u5e26\u5feb\u7167\u201d\uff0c\u7528 $a^H R a$ \u6784\u9020\u7a7a\u95f4\u8c31\u3002

\u53c2\u8003\u6587\u732e\u4e2d\u7684\u8868\u8fbe

$$
P_{CBF}(\theta)=a^H(u,\alpha,\theta)R(u,\alpha)a(u,\alpha,\theta)
$$

\u5c31\u5bf9\u5e94\u7b2c\u4e8c\u79cd\u65b9\u6cd5\u3002\u5b83\u7684\u597d\u5904\u662f\u5f62\u5f0f\u4e0a\u548c\u7a84\u5e26 CBF/MVDR/MUSIC \u5f88\u4e00\u81f4\uff0c\u540e\u7eed\u8981\u505a FRFT-MVDR \u6216 FRFT-MUSIC \u4e5f\u66f4\u5bb9\u6613\u8854\u63a5\u3002

\u5de5\u7a0b\u4e0a\u8981\u6700\u5c0f\u5fc3\u7684\u662f FRFT \u57df\u5bfc\u5411\u77e2\u91cf $a(u,\alpha,\theta)$\u3002\u5b83\u4e0d\u80fd\u7b80\u5355\u7528\u666e\u901a\u7a84\u5e26\u7684 $e^{-j2\pi f_c\tau_m}$ \u66ff\u4ee3\uff0c\u56e0\u4e3a FRFT \u4e0b\u7684\u5ef6\u65f6\u4f1a\u5e26\u6765\u66f4\u590d\u6742\u7684\u76f8\u4f4d\u548c bin \u504f\u79fb\u3002\u672c notebook \u7528\u201c\u6821\u51c6\u6d41\u5f62\u201d\u4fdd\u8bc1\u5bfc\u5411\u77e2\u91cf\u548c\u5f53\u524d DFRFT \u5b9e\u73b0\u4e00\u81f4\u3002
""",
    ),
]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nb["cells"] = [
        cell
        for cell in nb["cells"]
        if not str(cell.get("id", "")).startswith("frft-")
    ]
    nb["cells"].extend(frft_cells)
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(frft_cells)} FRFT cells. Total cells: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
