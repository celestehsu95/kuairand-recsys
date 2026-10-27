#!/usr/bin/env bash
# Runs the whole pipeline on a small fake dataset (~1-2 minutes on a laptop).
# Run this first: if it passes, the real run will too.
set -euo pipefail
cd "$(dirname "$0")"

python tests/make_synthetic.py --out_dir data/synthetic/KuaiRand-Pure/data
python prepare_data.py --raw_dir data/synthetic --out_dir data/processed_synth
python baselines.py --data_dir data/processed_synth --results_dir results_synth
python train_retrieval.py --data_dir data/processed_synth --results_dir results_synth --epochs 3 --batch_size 512
python train_retrieval.py --data_dir data/processed_synth --results_dir results_synth --epochs 3 --batch_size 512 --no_side_features
python train_retrieval.py --data_dir data/processed_synth --results_dir results_synth --epochs 3 --batch_size 512 --no_logq
python train_ranking.py --data_dir data/processed_synth --results_dir results_synth --model dcn --epochs 2
python train_ranking.py --data_dir data/processed_synth --results_dir results_synth --model dnn --epochs 2
python build_rt_features.py --data_dir data/processed_synth
python train_rt_ranker.py --data_dir data/processed_synth --results_dir results_synth --epochs 2
python export_artifacts.py --data_dir data/processed_synth --out_dir artifacts_synth
python make_report.py --results_dir results_synth

echo
echo "Smoke test passed. Now download the real data and run ./run_all.sh"
echo "(Optional) test the serving stack on fake data: ARTIFACTS=./artifacts_synth ./demo.sh"
