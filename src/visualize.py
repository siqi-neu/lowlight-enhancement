"""可视化：全部用 Pillow 绘制（不依赖 matplotlib）。

图 1  数据集与低照度退化
图 2  单图多方法对比 + 直方图
图 3  平均客观指标柱状图
图 4  局部放大
图 5  四张图的输入 / 本文方法 / 真值总览
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2gray

from .dataset import to_uint8

BG = (255, 255, 255)
INK = (17, 24, 39)
RED = (192, 57, 43)
GRAY = (156, 163, 175)
LIGHT = (209, 213, 219)

_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


@lru_cache(maxsize=32)
def font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(to_uint8(img))


def _fit(im: Image.Image, bw: int, bh: int) -> Image.Image:
    s = min(bw / im.width, bh / im.height)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)


def _figure(panels, cols: int, out: Path, suptitle: str, cell=(330, 300)):
    """panels: [(标题, 图像/PIL图, 副标题或None), ...]"""
    head, pad, cap, foot = 62, 14, 30, 34
    rows = (len(panels) + cols - 1) // cols
    cw, ch = cell[0] + 2 * pad, cell[1] + cap + foot + 2 * pad
    W, H = cols * cw, head + rows * ch + 8
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((W // 2, head // 2 + 4), suptitle, font=font(27), fill=INK, anchor="mm")

    for i, (title, img, sub) in enumerate(panels):
        c, r = i % cols, i // cols
        x0, y0 = c * cw, head + r * ch
        d.text((x0 + cw // 2, y0 + pad + cap // 2), title, font=font(21), fill=INK, anchor="mm")
        if img is not None:
            im = _fit(_pil(img), cell[0], cell[1])
            px = x0 + pad + (cell[0] - im.width) // 2
            py = y0 + pad + cap + (cell[1] - im.height) // 2
            canvas.paste(im, (px, py))
            d.rectangle([px - 1, py - 1, px + im.width, py + im.height], outline=LIGHT)
        if sub:
            d.text((x0 + cw // 2, y0 + pad + cap + cell[1] + foot // 2),
                   sub, font=font(17), fill=RED, anchor="mm")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


# --------------------------------------------------------------------------- #
def fig_dataset(samples: dict, out: Path):
    panels = []
    for name, (gt, low) in samples.items():
        panels.append((f"{name} · 真值 GT", gt, None))
    for name, (gt, low) in samples.items():
        panels.append((f"{name} · 低照度输入", low, None))
    _figure(panels, len(samples), out,
            "图 1  测试图与合成低照度退化  I = 0.72·J^2.4 + 高斯噪声", cell=(300, 300))


def _hist_panel(low: np.ndarray, out_img: np.ndarray, size=(330, 300)) -> Image.Image:
    W, H = size
    im = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(im)
    left, right, top, bottom = 34, 12, 18, 34
    gw, gh = W - left - right, H - top - bottom
    bins = 64
    h1, _ = np.histogram(rgb2gray(low).ravel(), bins=bins, range=(0, 1))
    h2, _ = np.histogram(rgb2gray(out_img).ravel(), bins=bins, range=(0, 1))
    mx = max(h1.max(), h2.max())

    # 坐标轴与网格
    for k in range(5):
        y = top + gh - gh * k / 4
        d.line([left, y, left + gw, y], fill=(238, 240, 243))
    d.line([left, top + gh, left + gw, top + gh], fill=GRAY)
    d.line([left, top, left, top + gh], fill=GRAY)

    def curve(h, color, width):
        pts = []
        for i, v in enumerate(h):
            x = left + gw * (i + 0.5) / bins
            y = top + gh - gh * (v / mx) ** 0.5  # 平方根压缩，暗部细节更可见
            pts.append((x, y))
        d.line(pts, fill=color, width=width, joint="curve")

    curve(h1, GRAY, 3)
    curve(h2, RED, 3)
    d.text((left, H - 22), "灰度", font=font(15), fill=INK)
    d.text((left + 4, top - 4), "像素数", font=font(15), fill=INK)
    # 图例
    d.line([left + gw - 92, top + 10, left + gw - 72, top + 10], fill=GRAY, width=3)
    d.text((left + gw - 66, top + 10), "输入", font=font(14), fill=INK, anchor="lm")
    d.line([left + gw - 92, top + 32, left + gw - 72, top + 32], fill=RED, width=3)
    d.text((left + gw - 66, top + 32), "本文", font=font(14), fill=INK, anchor="lm")
    return im


def fig_compare(name: str, low: np.ndarray, results: dict, gt: np.ndarray, scores: dict, out: Path):
    panels = [("低照度输入", low, "PSNR %.2f dB" % scores_input(low, gt))]
    for m, img in results.items():
        s = scores[m]
        panels.append((m, img, f"PSNR {s['PSNR(dB)']:.2f} dB · SSIM {s['SSIM']:.3f}"))
    panels.append(("真值 GT", gt, None))
    panels.append(("直方图对比", _hist_panel(low, results["本文方法"]), None))
    _figure(panels, 4, out, f"图 2  {name} 上的增强结果对比", cell=(300, 300))


def scores_input(low, gt):
    from .metrics import psnr
    return psnr(low, gt)


def fig_zoom(name: str, low: np.ndarray, results: dict, gt: np.ndarray, box, out: Path):
    y0, y1, x0, x1 = box
    items = [("低照度输入", low)] + list(results.items()) + [("真值 GT", gt)]
    panels = [(t, img[y0:y1, x0:x1], None) for t, img in items]
    _figure(panels, len(panels), out,
            f"图 4  {name} 局部放大（{y1 - y0}×{x1 - x0} 区域，看细节与偏色）", cell=(220, 220))


def fig_overview(samples: dict, all_results: dict, out: Path):
    panels = []
    for name, (gt, low) in samples.items():
        panels += [(f"{name} 输入", low, None),
                   (f"{name} 本文方法", all_results[name]["本文方法"], None),
                   (f"{name} 真值", gt, None)]
    _figure(panels, 3, out, "图 5  四张测试图：低照度输入 / 本文方法 / 真值", cell=(300, 300))


def fig_metrics(summary: dict, out: Path):
    methods = list(summary.keys())
    short = {"低照度输入(未处理)": "输入", "HE(逐通道)": "HE-RGB", "HE(仅V通道)": "HE-V",
             "CLAHE(L通道)": "CLAHE", "MSR(Retinex)": "MSR", "本文方法": "本文"}
    items = [("PSNR(dB)", "PSNR (dB) ↑"), ("SSIM", "SSIM ↑"), ("信息熵", "信息熵 ↑"),
             ("平均梯度", "平均梯度 ↑"), ("色度偏差", "色度偏差 ↓")]

    pw, ph, head = 330, 330, 66
    W, H = pw * 5, ph + head + 20
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((W // 2, 30), "图 3  四种测试图的平均客观指标（红 = 本文方法）",
           font=font(27), fill=INK, anchor="mm")

    for j, (key, label) in enumerate(items[:5]):
        x0 = j * pw
        vals = [summary[m][key] for m in methods]
        top_y, base_y = head + 46, head + ph - 70
        d.text((x0 + pw // 2, head + 22), label, font=font(19), fill=INK, anchor="mm")
        d.line([x0 + 42, base_y, x0 + pw - 22, base_y], fill=GRAY)
        vmax = max(vals) * 1.12
        bw = 30
        gap = (pw - 64 - bw * len(vals)) // max(1, len(vals) - 1)
        for i, (m, v) in enumerate(zip(methods, vals)):
            x = x0 + 50 + i * (bw + gap)
            h = (base_y - top_y) * (v / vmax)
            color = RED if m == "本文方法" else (GRAY if m != "MSR(Retinex)" else (120, 134, 150))
            d.rectangle([x, base_y - h, x + bw, base_y], fill=color)
            txt = f"{v:.2f}" if key in ("PSNR(dB)", "信息熵", "平均梯度") else f"{v:.3f}"
            d.text((x + bw / 2, base_y - h - 12), txt, font=font(13), fill=INK, anchor="mm")
            d.text((x + bw / 2, base_y + 14), short.get(m, m), font=font(13), fill=INK, anchor="mm")
    canvas.save(out)


def fig_ablation(rows, out: Path):
    """消融实验：每一步配置的 PSNR / SSIM / 色度偏差。"""
    labels = ["输入", "①CLAHE", "②gamma", "③细节", "④完整"]
    keys = [("PSNR(dB)", "PSNR (dB) ↑", 2), ("SSIM", "SSIM ↑", 4), ("色度偏差", "色度偏差 ↓", 2)]
    pw, ph, head = 430, 330, 66
    W, H = pw * len(keys), ph + head + 20
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((W // 2, 30), "图 6  消融实验：四个模块逐步叠加的效果（4 图平均）",
           font=font(26), fill=INK, anchor="mm")

    for j, (key, label, nd) in enumerate(keys):
        x0 = j * pw
        vals = [m[key] for _, m in rows]
        top_y, base_y = head + 50, head + ph - 64
        d.text((x0 + pw // 2, head + 22), label, font=font(19), fill=INK, anchor="mm")
        d.line([x0 + 44, base_y, x0 + pw - 20, base_y], fill=GRAY)
        vmax = max(vals) * 1.14
        bw, gap = 38, 26
        for i, (v, lab) in enumerate(zip(vals, labels)):
            x = x0 + 52 + i * (bw + gap)
            h = (base_y - top_y) * (v / vmax)
            d.rectangle([x, base_y - h, x + bw, base_y], fill=RED if i == len(vals) - 1 else (168, 180, 194))
            d.text((x + bw / 2, base_y - h - 12), f"{v:.{nd}f}", font=font(13), fill=INK, anchor="mm")
            d.text((x + bw / 2, base_y + 15), lab, font=font(11), fill=INK, anchor="mm")
    canvas.save(out)
