"""客观评价指标（对应课件第 2 章保真度准则 + 第 3 章增强效果评价）。"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray, rgb2lab
from skimage.measure import shannon_entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .dataset import to_uint8


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR = 10 * log10(MAX^2 / MSE)，单位 dB，越高越接近真值。"""
    return float(peak_signal_noise_ratio(gt, pred, data_range=1.0))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """结构相似度，兼顾亮度、对比度、结构，∈[-1,1]，越接近 1 越好。"""
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))


def entropy(img: np.ndarray) -> float:
    """一维灰度熵 H = -sum p*log2(p)，反映信息量/细节丰富度。"""
    return float(shannon_entropy(to_uint8(rgb2gray(img))))


def mean_lightness(img: np.ndarray) -> float:
    """CIE Lab 的 L* 均值（0~100），衡量整体亮度。"""
    return float(rgb2lab(img)[..., 0].mean())


def rms_contrast(img: np.ndarray) -> float:
    """RMS 对比度 = 灰度标准差。"""
    return float(rgb2gray(img).std())


def mean_gradient(img: np.ndarray) -> float:
    """平均梯度：反映边缘/细节清晰度，越大越锐利（也越大越可能放大噪声）。"""
    g = rgb2gray(img)
    gy, gx = np.gradient(g)
    return float(np.sqrt((gx ** 2 + gy ** 2) / 2.0).mean())


def chroma_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """色度偏差：Lab 中 (a,b) 平面的平均欧氏距离，越小说明色彩越忠实。"""
    pa = rgb2lab(pred)[..., 1:]
    ga = rgb2lab(gt)[..., 1:]
    return float(np.sqrt(((pa - ga) ** 2).sum(axis=2)).mean())


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    return {
        "PSNR(dB)": psnr(pred, gt),
        "SSIM": ssim(pred, gt),
        "信息熵": entropy(pred),
        "平均亮度L*": mean_lightness(pred),
        "RMS对比度": rms_contrast(pred),
        "平均梯度": mean_gradient(pred),
        "色度偏差": chroma_error(pred, gt),
    }
