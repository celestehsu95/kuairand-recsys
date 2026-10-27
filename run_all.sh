#!/usr/bin/env bash
# Full experiment on KuaiRand-Pure: preprocessing, baselines, two-tower (+ ablations), DCN-v2 (+ ablation), report.
set -euo pipefail
cd "$(dirname "$0")"

python prepare_data.py --raw_dir data/KuaiRand-Pure --out_dir data/processed
python baselines.py
python train_retrieval.py
python train_retrieval.py --no_side_features
python train_retrieval.py --no_logq
python train_ranking.py --model dcn
python train_ranking.py --model dnn
python build_rt_features.py
python train_rt_ranker.py
python export_artifacts.py
python make_report.py --update_readme
