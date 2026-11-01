"""图像增强算法集合（对应课件第 3 章：灰度变换 / 直方图 / 空域滤波 / Retinex）。

方法层级：
    he_rgb   基线   逐通道全局直方图均衡化（教材最朴素做法）
    he_v     对照   仅对 HSV 的 V 通道做全局直方图均衡化
    clahe    中间   仅对 LAB 的 L 通道做自适应直方图均衡化（CLAHE）
    msr      对照   多尺度 Retinex（S = R x I 理论）
    proposed 本文   CLAHE + 亮度自适应 gamma + 引导滤波细节增强 + 色度保护

仅依赖 numpy / scipy / scikit-image，便于复现。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.color import hsv2rgb, lab2rgb, rgb2hsv, rgb2lab
from skimage.exposure import equalize_adapthist, equalize_hist


# --------------------------------------------------------------------------- #
# 基础算子
# --------------------------------------------------------------------------- #
def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 16, eps: float = 1e-3) -> np.ndarray:
    """引导滤波（He et al.）：边缘保持的低通，用于把亮度图分解为 base + detail。"""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    size = 2 * radius + 1

    mean_i = uniform_filter(guide, size, mode="reflect")
    mean_p = uniform_filter(src, size, mode="reflect")
    var_i = uniform_filter(guide * guide, size, mode="reflect") - mean_i ** 2
    cov_ip = uniform_filter(guide * src, size, mode="reflect") - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return (uniform_filter(a, size, mode="reflect") * guide
            + uniform_filter(b, size, mode="reflect")).astype(np.float32)


def auto_gamma(luma: np.ndarray, target: float = 0.55, lo: float = 0.35, hi: float = 1.8) -> float:
    """按当前亮度反解 gamma，使输出亮度趋向 target（暗图用 gamma<1 提亮）。"""
    m = float(np.clip(np.mean(luma), 1e-3, 1.0))
    return float(np.clip(np.log(target) / np.log(m), lo, hi))


# --------------------------------------------------------------------------- #
# 方法 1：基线 —— 逐通道全局直方图均衡化
# --------------------------------------------------------------------------- #
def he_rgb(img: np.ndarray) -> np.ndarray:
    """对 R/G/B 三个通道各自做全局直方图均衡化。

    教材入门示例，但三通道的累积分布函数不同，均衡后各通道灰度映射不一致，
    容易引入明显色偏（本实验的改进动机之一）。
    """
    out = np.stack([equalize_hist(np.clip(img[..., c], 0, 1), nbins=256) for c in range(3)], axis=-1)
    return np.clip(out, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 方法 2：对照 —— 仅对亮度通道（HSV-V）做全局直方图均衡化
# --------------------------------------------------------------------------- #
def he_v(img: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(np.clip(img, 0, 1))
    hsv[..., 2] = equalize_hist(hsv[..., 2], nbins=256)
    return np.clip(hsv2rgb(hsv), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 方法 3：中间 —— LAB-L 通道 CLAHE
# --------------------------------------------------------------------------- #
def clahe_lab(img: np.ndarray, clip_limit: float = 0.01, tile: int = 8) -> np.ndarray:
    """clip_limit 采用 skimage 的 [0,1] 归一化定义，0.01 约等于 OpenCV 的 clipLimit=2.0。"""
    lab = rgb2lab(np.clip(img, 0, 1))
    L = np.clip(lab[..., 0] / 100.0, 0, 1)
    kernel = max(2, img.shape[0] // tile)
    L = equalize_adapthist(L, kernel_size=kernel, clip_limit=clip_limit, nbins=256)
    lab[..., 0] = L * 100.0
    return np.clip(lab2rgb(lab), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 方法 4：对照 —— 多尺度 Retinex (MSR / MSRCR 简化实现)
# --------------------------------------------------------------------------- #
def msr(img: np.ndarray, sigmas: tuple[float, ...] = (15.0, 80.0, 250.0), target: float = 0.45) -> np.ndarray:
    """多尺度 Retinex：S = R x I，取对数后 R = log S - log(G * S)，三个尺度加权平均。

    输出的反射分量取指数后按中位数归一到目标亮度（Retinex 只保证相对亮度关系，
    绝对亮度需要一次自适应增益，否则结果会整体过曝或偏暗）。
    """
    from scipy.ndimage import gaussian_filter

    eps = 1e-6
    img = np.clip(img.astype(np.float32), eps, 1.0)
    log_img = np.log(img)
    retinex = np.zeros_like(img)
    for s in sigmas:
        blur = np.stack([gaussian_filter(img[..., c], s, mode="reflect") for c in range(3)], axis=-1)
        retinex += (log_img - np.log(np.clip(blur, eps, 1.0))) / len(sigmas)

    refl = np.exp(retinex)                       # 反射分量（相对亮度）
    refl = refl / (np.median(refl) + eps) * target
    return np.clip(refl, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 方法 5：本文方法 —— CLAHE + 自适应 gamma + 引导滤波细节增强 + 色度保护
# --------------------------------------------------------------------------- #
def proposed(
    img: np.ndarray,
    clip_limit: float = 0.01,
    tile: int = 8,
    detail_gain: float = 1.45,
    radius: int = 16,
    eps: float = 1e-3,
    sat_gain: float = 1.15,
    target: float = 0.55,
    use_gamma: bool = True,
    use_detail: bool = True,
    use_sat: bool = True,
) -> np.ndarray:
    """只在亮度通道上做增强，色度通道原样保留，从而避免偏色。

    步骤：
      1) RGB -> CIE Lab，取 L（亮度）通道，与色度 a/b 解耦；
      2) L 上做 CLAHE：分块自适应均衡，提升局部对比度，避免全局 HE 的过增强与噪声放大；
      3) 亮度自适应 gamma：按当前亮度反解 gamma，把整体亮度拉到目标区间；
      4) 引导滤波把 L 分解为 base + detail，对 detail 乘增益，锐化纹理而不放大噪声；
      5) 回写 L（a/b 保持），转回 RGB，并做轻微饱和度补偿。

    use_* 开关用于消融实验（见 src/ablation.py）。
    """
    lab = rgb2lab(np.clip(img, 0, 1))
    L = np.clip(lab[..., 0] / 100.0, 0, 1)

    # 2) CLAHE
    kernel = max(2, img.shape[0] // tile)
    L1 = equalize_adapthist(L, kernel_size=kernel, clip_limit=clip_limit, nbins=256)

    # 3) 自适应 gamma
    L2 = np.power(np.clip(L1, 0, 1), auto_gamma(L1, target=target)) if use_gamma else L1

    # 4) 引导滤波细节增强
    if use_detail:
        base = guided_filter(L2, L2, radius=radius, eps=eps)
        detail = L2 - base
        L3 = np.clip(base + detail_gain * detail, 0, 1)
    else:
        L3 = np.clip(L2, 0, 1)

    lab[..., 0] = L3 * 100.0
    out = np.clip(lab2rgb(lab), 0, 1)

    # 5) 饱和度补偿（亮度提升会带来视觉上的「褪色」）
    if use_sat:
        hsv = rgb2hsv(out)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_gain, 0, 1)
        out = hsv2rgb(hsv)
    return np.clip(out, 0, 1).astype(np.float32)


METHODS = {
    "HE(逐通道)": he_rgb,
    "HE(仅V通道)": he_v,
    "CLAHE(L通道)": clahe_lab,
    "MSR(Retinex)": msr,
    "本文方法": proposed,
}
