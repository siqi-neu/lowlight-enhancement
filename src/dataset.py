"""数据加载与低照度退化合成。

数据来源：scikit-image 内置标准测试图（免下载、可复现）。
退化模型：I = gain * J^gamma + n ，J 为真值图像，n 为传感器噪声。
"""

from __future__ import annotations

import zlib

import numpy as np
from skimage import data, transform

# 4 张公开标准测试图（scikit-image 自带，无需联网），覆盖人像/暗光静物/结构烟雾/医学眼底
IMAGE_SOURCES = {
    "astronaut": data.astronaut,
    "coffee": data.coffee,
    "rocket": data.rocket,
    "retina": data.retina,
}


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1]。"""
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def load_image(name: str, max_side: int = 512) -> np.ndarray:
    """读取标准图并统一长边到 max_side，返回 float32 RGB [0,1]。"""
    img = IMAGE_SOURCES[name]()
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = transform.resize(
            img, (int(round(h * scale)), int(round(w * scale)), 3),
            anti_aliasing=True, preserve_range=True,
        )
    img = np.clip(img, 0, 255)
    return to_float(img)


def seed_for(name: str) -> int:
    """名称 -> 固定随机种子（用 crc32 而非内置 hash，保证跨进程可复现）。"""
    return zlib.crc32(name.encode("utf-8")) % 1000


def synthesize_lowlight(
    img: np.ndarray,
    gamma: float = 2.4,
    gain: float = 0.72,
    noise_sigma: float = 0.012,
    seed: int = 0,
) -> np.ndarray:
    """合成低照度图像：gamma 压缩动态范围（模拟光照不足），再叠加高斯噪声。"""
    rng = np.random.default_rng(seed)
    low = gain * np.power(np.clip(img, 0, 1), gamma)
    low = low + rng.normal(0.0, noise_sigma, low.shape).astype(np.float32)
    return np.clip(low, 0.0, 1.0).astype(np.float32)
