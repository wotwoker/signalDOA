import fs from "node:fs";

const notebookPath = "code/experiments/LFM.ipynb";

function block(strings, ...values) {
  let text = String.raw({ raw: strings.raw }, ...values);
  text = text.replace(/^\n/, "").replace(/\s+$/, "");
  const lines = text.split("\n");
  const indents = lines
    .filter((line) => line.trim().length > 0)
    .map((line) => line.match(/^ */)[0].length);
  const minIndent = indents.length ? Math.min(...indents) : 0;
  return lines.map((line) => line.slice(minIndent)).join("\n");
}

function sourceLines(text) {
  return text.split("\n").map((line) => `${line}\n`);
}

function md(id, text) {
  return {
    cell_type: "markdown",
    id,
    metadata: {},
    source: sourceLines(text),
  };
}

function code(id, text) {
  return {
    cell_type: "code",
    execution_count: null,
    id,
    metadata: {},
    outputs: [],
    source: sourceLines(text),
  };
}

const frftCells = [
  md(
    "frft-overview",
    block`
      # 用 FRFT 分析 LFM 信号

      前面已经从时域、频域和时频图观察了 LFM。这里重新整理 FRFT 部分，只保留一条主线：

      1. LFM 是二次相位信号。
      2. FRFT 可以理解为时频平面的旋转。
      3. 当旋转角度与 chirp 斜率匹配时，LFM 能量会在某个分数阶域聚焦。
      4. 聚焦后的分量可以作为后续 LFM 检测、参数估计或宽带 DOA 处理的特征。

      LFM 的复解析形式为

      $$
      s(t)=A\exp\left[j2\pi\left(f_0t+\frac{1}{2}\mu t^2\right)+j\phi\right]
      $$

      其中二次相位项 $\pi\mu t^2$ 决定了时频轨迹的斜率。FRFT 的阶次记为 $p$，旋转角为

      $$
      \alpha=p\frac{\pi}{2}
      $$

      一般连续形式可写成

      $$
      X_\alpha(u)=\int_{-\infty}^{\infty}x(t)K_\alpha(t,u)\,dt
      $$

      当 $\alpha$ 不是 $k\pi$ 时，核函数常写为

      $$
      K_\alpha(t,u)=A_\alpha
      \exp\left[j\pi\left(t^2\cot\alpha-2tu\csc\alpha+u^2\cot\alpha\right)\right]
      $$

      可以看到 FRFT 核函数也包含二次相位。因此它对 LFM 很自然：阶次合适时，chirp 的二次相位被更好地匹配，能量由“斜线分布”变成“尖峰聚焦”。

      工程上不要直接死记某个最佳阶次公式。离散 FRFT 的尺度、采样长度和归一化方式会改变数值关系，所以这里采用阶次扫描来寻找最聚焦的 $p$。
    `,
  ),
  code(
    "frft-discrete-implementation",
    block`
      # -----------------------------
      # 5. Discrete FRFT implementation
      # -----------------------------
      # This direct O(N^2) DFRFT is used for clarity. We downsample the LFM
      # segment so the order scan remains interactive in a notebook.

      def dft_unitary(x):
          """Unitary DFT with centered frequency bins."""
          return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x))) / np.sqrt(len(x))


      def idft_unitary(x):
          """Unitary inverse DFT with centered frequency bins."""
          return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x))) * np.sqrt(len(x))


      def frft_direct(x, p):
          """Direct discrete FRFT approximation.

          p is the fractional order and alpha = p*pi/2.
          The special cases p=0,1,2,3 are handled explicitly.

          This is a pedagogical DFRFT convention using dimensionless centered
          sample coordinates. It is suitable for visual analysis and order
          scanning, but not meant to be a fast production FRFT.
          """
          x = np.asarray(x, dtype=complex)
          N_local = x.size
          p_mod = np.mod(p, 4.0)

          if np.isclose(p_mod, 0.0):
              return x.copy()
          if np.isclose(p_mod, 1.0):
              return dft_unitary(x)
          if np.isclose(p_mod, 2.0):
              return x[::-1].copy()
          if np.isclose(p_mod, 3.0):
              return idft_unitary(x)

          alpha = p_mod * np.pi / 2
          sin_alpha = np.sin(alpha)
          cot_alpha = 1 / np.tan(alpha)
          csc_alpha = 1 / sin_alpha

          n = np.arange(N_local) - (N_local - 1) / 2
          t_norm = n / np.sqrt(N_local)
          u_norm = n / np.sqrt(N_local)
          T, U = np.meshgrid(t_norm, u_norm, indexing="ij")

          phase_kernel = np.pi * (
              T**2 * cot_alpha
              - 2 * T * U * csc_alpha
              + U**2 * cot_alpha
          )
          kernel = np.exp(1j * phase_kernel)
          return kernel.conj().T @ x / np.sqrt(N_local * abs(sin_alpha))


      frft_stride = 4
      s_frft = s[::frft_stride]
      t_frft = t[::frft_stride]
      fs_frft = fs / frft_stride
      N_frft = s_frft.size
      s_frft_input = s_frft / np.sqrt(np.sum(np.abs(s_frft) ** 2))

      frft_p0 = frft_direct(s_frft_input, 0)
      frft_p1 = frft_direct(s_frft_input, 1)
      frft_p2 = frft_direct(s_frft_input, 2)

      print("FRFT analysis segment")
      print(f"N_frft = {N_frft}, fs_frft = {fs_frft / 1e3:.1f} kHz, stride = {frft_stride}")
      print("FRFT sanity checks")
      print(f"p=0 reconstruction error = {np.linalg.norm(frft_p0 - s_frft_input):.2e}")
      print(f"p=1 spectrum length = {frft_p1.size}")
      print(f"p=2 time-reversal error = {np.linalg.norm(frft_p2 - s_frft_input[::-1]):.2e}")
    `,
  ),
  md(
    "frft-order-scan-note",
    block`
      ## 阶次扫描与聚焦指标

      对每个候选阶次 $p$ 计算 FRFT，然后用一个简单指标判断能量是否聚焦：

      $$
      \eta(p)=\frac{\max_u |X_p(u)|^2}{\mathrm{mean}_u |X_p(u)|^2}
      $$

      如果能量摊得很开，最大值不会比平均值大太多；如果能量被压成尖峰，$\eta(p)$ 会明显升高。下面避开 $p=0$ 和 $p=2$ 的奇异点，在 $(0,2)$ 内扫描。
    `,
  ),
  code(
    "frft-order-scan",
    block`
      # -----------------------------
      # 6. Scan fractional order and find strongest concentration
      # -----------------------------
      orders = np.linspace(0.05, 1.95, 121)
      frft_peak_to_mean = np.zeros_like(orders)
      frft_entropy = np.zeros_like(orders)

      for i, p in enumerate(orders):
          Xp = frft_direct(s_frft_input, p)
          power_p = np.abs(Xp) ** 2
          power_p = power_p / np.sum(power_p)

          frft_peak_to_mean[i] = np.max(power_p) / np.mean(power_p)
          frft_entropy[i] = -np.sum(power_p * np.log(power_p + np.finfo(float).tiny))

      best_index = int(np.argmax(frft_peak_to_mean))
      p_best = float(orders[best_index])
      alpha_best = p_best * np.pi / 2

      print(f"Best FRFT order by peak-to-mean = {p_best:.3f}")
      print(f"Best rotation angle alpha = {np.rad2deg(alpha_best):.2f} deg")
      print(f"Peak-to-mean at best order = {frft_peak_to_mean[best_index]:.2f}")
      print(f"Entropy at best order = {frft_entropy[best_index]:.2f}")

      plt.figure(figsize=(10, 4))
      plt.plot(orders, frft_peak_to_mean, linewidth=1.8, label="peak-to-mean")
      plt.axvline(p_best, color="tab:red", linestyle="--", linewidth=1.2, label=f"best p = {p_best:.3f}")
      plt.xlabel("FRFT order p")
      plt.ylabel("Concentration metric")
      plt.title("FRFT order scan for LFM concentration")
      plt.legend(loc="best")
      plt.tight_layout()
    `,
  ),
  code(
    "frft-fft-vs-frft",
    block`
      # -----------------------------
      # 7. Compare ordinary FFT and best-order FRFT
      # -----------------------------
      X_best = frft_direct(s_frft_input, p_best)
      X_fft = frft_direct(s_frft_input, 1.0)

      u_axis = np.arange(N_frft) - (N_frft - 1) / 2
      freq_fft = np.fft.fftshift(np.fft.fftfreq(N_frft, d=1 / fs_frft)) / 1e3

      X_best_db = 20 * np.log10(
          np.maximum(np.abs(X_best), np.finfo(float).tiny) / np.max(np.abs(X_best))
      )
      X_fft_db = 20 * np.log10(
          np.maximum(np.abs(X_fft), np.finfo(float).tiny) / np.max(np.abs(X_fft))
      )

      focus_bin = int(np.argmax(np.abs(X_best)))

      plt.figure(figsize=(11, 5))

      plt.subplot(1, 2, 1)
      plt.plot(freq_fft, X_fft_db, linewidth=1.5)
      plt.axvline(f0 / 1e3, color="tab:red", linestyle="--", linewidth=0.8, label="start f")
      plt.axvline((f0 + mu * duration) / 1e3, color="tab:green", linestyle="--", linewidth=0.8, label="end f")
      plt.xlim(0, 60)
      plt.ylim(-60, 3)
      plt.xlabel("Frequency (kHz)")
      plt.ylabel("Normalized magnitude (dB)")
      plt.title("Ordinary FFT, p = 1")
      plt.legend(loc="upper right")

      plt.subplot(1, 2, 2)
      plt.plot(u_axis, X_best_db, linewidth=1.5, color="tab:purple")
      plt.axvline(u_axis[focus_bin], color="tab:red", linestyle="--", linewidth=0.9, label="focused bin")
      plt.ylim(-60, 3)
      plt.xlabel("FRFT-domain sample index")
      plt.ylabel("Normalized magnitude (dB)")
      plt.title(f"Best FRFT domain, p = {p_best:.3f}")
      plt.legend(loc="upper right")
      plt.tight_layout()
    `,
  ),
  md(
    "frft-cbf-note",
    block`
      ## FRFT-CBF：保留一个清晰的 DOA 示例

      对阵列 LFM 信号，最稳妥的教学流程是先做宽带延时对齐，再做 FRFT 聚焦：

      $$
      y_\theta(t)=\frac{1}{M}\sum_{m=0}^{M-1}x_m(t+\tau_m(\theta)),
      \quad
      \tau_m(\theta)=\frac{p_m\sin\theta}{c}
      $$

      如果候选角 $\theta$ 接近真实 DOA，各阵元的 chirp 会在时间上对齐，求和后能量更强。随后对 $y_\theta(t)$ 做最佳阶次 FRFT，并在同一个聚焦 bin 上比较功率：

      $$
      P(\theta)=\left|\mathcal F^{p_\star}\{y_\theta(t)\}(u_0)\right|^2
      $$

      这里有一个重要边界：不要把普通窄带 steering vector 直接搬到 FRFT 域里。时间延时在 FRFT 域不只是简单相位，还会牵涉分数阶域坐标偏移和二次相位项。下面代码只保留 delay-sum + fixed focused bin 这条清楚、可验证的流程。
    `,
  ),
  code(
    "frft-cbf-simulation",
    block`
      # -----------------------------
      # 8. ULA LFM array simulation for FRFT-CBF
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


      theta_true_rad = np.deg2rad(theta_true_lfm_deg)
      tau_true = positions_lfm * np.sin(theta_true_rad) / c_lfm

      X_lfm_clean = np.vstack([analytic_lfm_at(t - tau_m) for tau_m in tau_true])
      X_lfm = add_complex_awgn_matrix(X_lfm_clean, snr_db=snr_lfm_db, seed=2026)

      print("FRFT-CBF array simulation")
      print(f"M = {M_lfm}, d = {d_lfm:.4f} m")
      print(f"true DOA = {theta_true_lfm_deg:.1f} deg, SNR = {snr_lfm_db:.1f} dB")
      print(f"X_lfm shape = {X_lfm.shape}  # (sensors, samples)")
      print(f"max inter-sensor delay = {np.max(np.abs(tau_true)) * 1e6:.2f} us")
    `,
  ),
  code(
    "frft-cbf-spectrum",
    block`
      # -----------------------------
      # 9. Delay-sum then FRFT focused spatial spectrum
      # -----------------------------
      def interp_complex_1d(t_axis, x, t_query):
          """Linear interpolation for complex-valued sampled data."""
          real = np.interp(t_query, t_axis, np.real(x), left=0.0, right=0.0)
          imag = np.interp(t_query, t_axis, np.imag(x), left=0.0, right=0.0)
          return real + 1j * imag


      def frft_matrix_direct(N_local, p):
          """Matrix form matching frft_direct(x, p), useful for angle scanning."""
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
          phase_kernel = np.pi * (
              T**2 * cot_alpha
              - 2 * T * U * csc_alpha
              + U**2 * cot_alpha
          )
          kernel = np.exp(1j * phase_kernel)
          return kernel.conj().T / np.sqrt(N_local * abs(sin_alpha))


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


      scan_angles_frft_cbf = np.arange(-60.0, 60.0 + 0.1, 0.2)

      max_scan_delay = (
          np.max(np.abs(positions_lfm))
          * np.sin(np.deg2rad(np.max(np.abs(scan_angles_frft_cbf))))
          / c_lfm
      )
      t_cbf_all = t[::frft_stride]
      valid_cbf = (t_cbf_all >= max_scan_delay) & (t_cbf_all <= duration - max_scan_delay)
      t_cbf = t_cbf_all[valid_cbf]
      N_cbf = t_cbf.size

      # Re-estimate the best order for the guarded CBF window.
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

      frft_cbf_power_matrix = np.zeros((scan_angles_frft_cbf.size, N_cbf))

      for i, theta_deg in enumerate(scan_angles_frft_cbf):
          theta_rad = np.deg2rad(theta_deg)
          tau_scan = positions_lfm * np.sin(theta_rad) / c_lfm

          aligned_sum = np.zeros(N_cbf, dtype=complex)
          for m in range(M_lfm):
              aligned_sum += interp_complex_1d(t, X_lfm[m], t_cbf + tau_scan[m])

          beam_output = aligned_sum / M_lfm
          beam_frft = F_cbf_best @ beam_output
          frft_cbf_power_matrix[i] = np.abs(beam_frft) ** 2

      global_angle_index, global_focus_bin = np.unravel_index(
          np.argmax(frft_cbf_power_matrix),
          frft_cbf_power_matrix.shape,
      )
      frft_cbf_power = frft_cbf_power_matrix[:, global_focus_bin]
      frft_cbf_power_maxbin = np.max(frft_cbf_power_matrix, axis=1)

      theta_hat_frft_cbf_grid = float(scan_angles_frft_cbf[np.argmax(frft_cbf_power)])
      theta_hat_frft_cbf = parabolic_peak_interpolation(scan_angles_frft_cbf, frft_cbf_power)

      frft_cbf_power_db = 10 * np.log10(
          np.maximum(frft_cbf_power, np.finfo(float).tiny) / np.max(frft_cbf_power)
      )
      frft_cbf_power_maxbin_db = 10 * np.log10(
          np.maximum(frft_cbf_power_maxbin, np.finfo(float).tiny) / np.max(frft_cbf_power_maxbin)
      )

      print(f"CBF valid samples = {N_cbf}, guard = {max_scan_delay * 1e6:.2f} us")
      print(f"full-pulse p_best = {p_best:.3f}, CBF-window p_best = {p_cbf_best:.3f}")
      print(f"global FRFT focus bin = {global_focus_bin}")
      print(f"FRFT-CBF grid estimate = {theta_hat_frft_cbf_grid:.2f} deg")
      print(f"FRFT-CBF interpolated estimate = {theta_hat_frft_cbf:.2f} deg")
      print(f"absolute error = {abs(theta_hat_frft_cbf - theta_true_lfm_deg):.2f} deg")

      plt.figure(figsize=(10, 4.5))
      plt.plot(
          scan_angles_frft_cbf,
          frft_cbf_power_maxbin_db,
          color="tab:blue",
          linestyle="--",
          linewidth=1.1,
          alpha=0.55,
          label="max over FRFT bins",
      )
      plt.plot(
          scan_angles_frft_cbf,
          frft_cbf_power_db,
          color="tab:red",
          linewidth=1.8,
          label="fixed focused FRFT bin",
      )
      plt.axvline(theta_true_lfm_deg, color="tab:green", linestyle="--", linewidth=1.2, label=f"true {theta_true_lfm_deg:.1f} deg")
      plt.axvline(theta_hat_frft_cbf, color="k", linestyle=":", linewidth=1.8, label=f"estimated {theta_hat_frft_cbf:.2f} deg")
      plt.ylim(-40, 3)
      plt.xlim(scan_angles_frft_cbf[0], scan_angles_frft_cbf[-1])
      plt.xlabel("Scan angle (deg)")
      plt.ylabel("Normalized focused FRFT power (dB)")
      plt.title("FRFT-CBF DOA estimation for LFM")
      plt.legend(loc="best")
      plt.tight_layout()
    `,
  ),
  md(
    "frft-summary",
    block`
      ## 小结

      这一节的 FRFT 逻辑可以压缩成三句话：

      1. LFM 的二次相位让它在时频平面上表现为斜线。
      2. FRFT 等价于换一个旋转角度观察信号；角度匹配时，LFM 能量会聚焦。
      3. 对阵列信号，先用宽带 delay-sum 做空间对齐，再用 FRFT 聚焦，是一个清楚且不容易误用窄带假设的流程。

      常用性质：

      - 周期性：$\mathcal{F}^{p+4}\{x\}=\mathcal{F}^{p}\{x\}$。
      - 逆变换：$(\mathcal{F}^{p})^{-1}=\mathcal{F}^{-p}$。
      - 可加性：理想连续定义下 $\mathcal{F}^{p_1}\{\mathcal{F}^{p_2}\{x\}\}=\mathcal{F}^{p_1+p_2}\{x\}$。
      - 聚焦性：不同 chirp 斜率通常对应不同的最佳 FRFT 阶次。

      工程注意：离散 FRFT 有多种实现和归一化约定。本 notebook 的实现用于教学和可视化；如果后续要做大规模处理、精确参数估计，或复现论文里的 $a^H R a$ 形式，需要先统一 FRFT 定义、时间/频率尺度和 FRFT 域 steering vector 标定。
    `,
  ),
];

const notebook = JSON.parse(fs.readFileSync(notebookPath, "utf8"));
const firstFrftIndex = notebook.cells.findIndex((cell) => String(cell.id || "").startsWith("frft-"));
const cellsBeforeFrft = notebook.cells.filter((cell) => !String(cell.id || "").startsWith("frft-"));

if (firstFrftIndex === -1) {
  cellsBeforeFrft.push(...frftCells);
  notebook.cells = cellsBeforeFrft;
} else {
  notebook.cells = [
    ...notebook.cells.slice(0, firstFrftIndex).filter((cell) => !String(cell.id || "").startsWith("frft-")),
    ...frftCells,
    ...notebook.cells.slice(firstFrftIndex).filter((cell) => !String(cell.id || "").startsWith("frft-")),
  ];
}

fs.writeFileSync(notebookPath, `${JSON.stringify(notebook, null, 2)}\n`, "utf8");
console.log(`Replaced FRFT section with ${frftCells.length} organized cells. Total cells: ${notebook.cells.length}`);
