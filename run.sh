#!/usr/bin/env bash
# 一键复现：主实验 + 消融实验
set -e
cd "$(dirname "$0")"

python -m src.run_experiment
python -m src.ablation

echo "完成：results/metrics.csv, results/ablation.md, results/figures/*.png"
