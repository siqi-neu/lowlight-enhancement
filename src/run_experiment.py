"""主实验脚本：合成低照度数据 -> 多方法增强 -> 客观指标 -> 图表落盘。

用法：
    cd lowlight-enhancement
    python -m src.run_experiment
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .dataset import IMAGE_SOURCES, seed_for, synthesize_lowlight, to_uint8
from .enhance import METHODS
from .metrics import evaluate
from . import visualize as viz
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "results" / "figures"
RES_DIR = ROOT / "results"

# 局部放大区域 (y0, y1, x0, x1)，选在细节/边缘密集处
ZOOM_BOX = {
    "astronaut": (150, 330, 220, 400),
    "coffee": (100, 280, 180, 360),
    "rocket": (60, 240, 250, 430),
    "retina": (150, 330, 150, 330),
}


def save_png(path: Path, img: np.ndarray):
    Image.fromarray(to_uint8(img)).save(path)


def main():
    for d in (DATA_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    names = list(IMAGE_SOURCES.keys())
    rows = []          # 逐 (图, 方法) 的指标
    summary = {}       # 方法 -> 平均指标
    samples, all_results, all_scores = {}, {}, {}

    for name in names:
        gt = _load(name)
        low = synthesize_lowlight(gt, seed=seed_for(name))
        samples[name] = (gt, low)
        save_png(DATA_DIR / f"{name}_gt.png", gt)
        save_png(DATA_DIR / f"{name}_low.png", low)

        results, scores = {}, {}
        # 参考：不处理的低照度输入
        base = evaluate(low, gt)
        base["耗时(ms)"] = 0.0
        rows.append({"图像": name, "方法": "低照度输入(未处理)", **base})
        for method, fn in METHODS.items():
            t0 = time.perf_counter()
            out = fn(low)
            cost = (time.perf_counter() - t0) * 1000.0
            m = evaluate(out, gt)
            m["耗时(ms)"] = cost
            rows.append({"图像": name, "方法": method, **m})
            results[method] = out
            scores[method] = m
            print(f"[{name}] {method:<14} PSNR={m['PSNR(dB)']:5.2f} SSIM={m['SSIM']:.3f} "
                  f"熵={m['信息熵']:.2f} L*={m['平均亮度L*']:5.1f} {cost:6.1f}ms")

        all_results[name], all_scores[name] = results, scores
        viz.fig_compare(name, low, results, gt, scores, FIG_DIR / f"fig2_compare_{name}.png")
        viz.fig_zoom(name, low, results, gt, ZOOM_BOX[name], FIG_DIR / f"fig4_zoom_{name}.png")

    viz.fig_dataset(samples, FIG_DIR / "fig1_dataset.png")
    viz.fig_overview(samples, all_results, FIG_DIR / "fig5_overview.png")

    # ---- 汇总（4 图平均） ----
    metric_keys = ["PSNR(dB)", "SSIM", "信息熵", "平均亮度L*", "RMS对比度", "平均梯度", "色度偏差", "耗时(ms)"]
    methods_order = ["低照度输入(未处理)"] + list(METHODS.keys())
    for m in methods_order:
        vals = [r for r in rows if r["方法"] == m]
        summary[m] = {k: float(np.mean([v[k] for v in vals])) for k in metric_keys}

    viz.fig_metrics(summary, FIG_DIR / "fig3_metrics.png")

    # ---- 落盘 ----
    _write_csv(rows, metric_keys, RES_DIR / "metrics.csv")
    _write_md(summary, methods_order, metric_keys, RES_DIR / "metrics.md")
    (RES_DIR / "summary.json").write_text(
        json.dumps({"summary": summary, "detail": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 4 张测试图平均 =====")
    _print_table(summary, methods_order, metric_keys)


def _load(name):
    from .dataset import load_image
    return load_image(name)


def _write_csv(rows, keys, path):
    cols = ["图像", "方法"] + keys
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join([str(r["图像"]), r["方法"]] + [f"{r[k]:.4f}" for k in keys]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(k, v):
    return f"{v:.2f}" if k in ("PSNR(dB)", "信息熵", "平均亮度L*", "平均梯度", "耗时(ms)") else f"{v:.4f}"


def _write_md(summary, order, keys, path):
    head = "| 方法 | " + " | ".join(keys) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(keys)) + " |"
    lines = [head, sep]
    for m in order:
        lines.append(f"| {m} | " + " | ".join(_fmt(k, summary[m][k]) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_table(summary, order, keys):
    print("方法".ljust(22) + "".join(k.rjust(12) for k in keys))
    for m in order:
        print(m.ljust(22) + "".join(_fmt(k, summary[m][k]).rjust(12) for k in keys))


if __name__ == "__main__":
    main()
