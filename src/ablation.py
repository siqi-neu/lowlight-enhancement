"""消融实验：逐步开启本文方法的四个模块，看每一步贡献多少。

用法：
    python -m src.ablation
"""

from __future__ import annotations

import numpy as np

from .dataset import IMAGE_SOURCES, load_image, seed_for, synthesize_lowlight
from .enhance import proposed
from .metrics import evaluate
from .visualize import fig_ablation  # 复用 Pillow 绘图
from .run_experiment import RES_DIR, FIG_DIR

STEPS = [
    ("① 仅 CLAHE（亮度通道）", dict(use_gamma=False, use_detail=False, use_sat=False)),
    ("② + 自适应 gamma", dict(use_gamma=True, use_detail=False, use_sat=False)),
    ("③ + 引导滤波细节增强", dict(use_gamma=True, use_detail=True, use_sat=False)),
    ("④ + 饱和度补偿（完整）", dict(use_gamma=True, use_detail=True, use_sat=True)),
]
KEYS = ["PSNR(dB)", "SSIM", "信息熵", "平均亮度L*", "平均梯度", "色度偏差"]


def main():
    data = []
    for name in IMAGE_SOURCES:
        gt = load_image(name)
        data.append((gt, synthesize_lowlight(gt, seed=seed_for(name))))

    # 参照：原始输入的指标
    base = {k: float(np.mean([evaluate(low, gt)[k] for gt, low in data])) for k in KEYS}
    rows = [("⓪ 低照度输入（不处理）", base)]
    for label, kw in STEPS:
        vals = {k: [] for k in KEYS}
        for gt, low in data:
            m = evaluate(proposed(low, **kw), gt)
            for k in KEYS:
                vals[k].append(m[k])
        rows.append((label, {k: float(np.mean(v)) for k, v in vals.items()}))
        print(f"{label:<20} " + "  ".join(f"{k}={np.mean(vals[k]):.3f}" for k in KEYS))

    lines = ["| 配置 | " + " | ".join(KEYS) + " |", "| --- | " + " | ".join(["---"] * len(KEYS)) + " |"]
    for label, m in rows:
        cells = [f"{m[k]:.4f}" if k == "SSIM" else
                 (f"{m[k]:.3f}" if k == "平均梯度" else f"{m[k]:.2f}") for k in KEYS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    (RES_DIR / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig_ablation([(l, m) for l, m in rows], FIG_DIR / "fig6_ablation.png")
    print("\n已写入 results/ablation.md 与 results/figures/fig6_ablation.png")


if __name__ == "__main__":
    main()
