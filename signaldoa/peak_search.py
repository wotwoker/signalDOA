"""空间谱峰值搜索工具。"""

from __future__ import annotations

import numpy as np


def find_top_peaks(
    scan_angles: np.ndarray,
    power: np.ndarray,
    num_sources: int,
    min_separation_deg: float | None = None,
    min_relative_height_db: float = -10.0,
    interpolate: bool = True,
) -> np.ndarray:
    """从空间谱中找出若干个相互分开的强峰。

    默认用扫描步长的 2 倍作为最小峰间隔，只防止同一个局部峰被重复选择。

    如果空间谱里两个源已经合并成一个主瓣，这个函数不会强行返回两个相邻点。
    那种情况下应该认为当前波束形成方法没有分辨出两个源，而不是把旁瓣误判
    成第二个源。
    """
    scan_step = float(np.median(np.diff(scan_angles)))
    if min_separation_deg is None:
        min_separation_deg = 2 * abs(scan_step)

    peak_indices = np.flatnonzero((power[1:-1] >= power[:-2]) & (power[1:-1] >= power[2:])) + 1

    if peak_indices.size == 0:
        peak_indices = np.array([np.argmax(power)])

    power_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny) / np.max(power))
    peak_indices = peak_indices[power_db[peak_indices] >= min_relative_height_db]
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

    if len(selected) < num_sources:
        print(
            f"Warning: only found {len(selected)} separated peak(s). "
            "The spectrum may not resolve these close sources."
        )

    peak_angles = []
    for index in selected:
        angle = scan_angles[index]

        # 三点抛物线插值可以减小扫描网格造成的读数误差。
        if interpolate and 0 < index < len(power) - 1:
            y_left = power[index - 1]
            y_mid = power[index]
            y_right = power[index + 1]
            denominator = y_left - 2 * y_mid + y_right
            if abs(denominator) > np.finfo(float).eps:
                offset = 0.5 * (y_left - y_right) / denominator
                offset = np.clip(offset, -1.0, 1.0)
                angle = angle + offset * scan_step

        peak_angles.append(angle)

    return np.sort(np.array(peak_angles))
