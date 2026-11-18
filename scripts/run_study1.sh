#!/usr/bin/env bash
# Every GPU step of Study 1, in order. Each stage is resumable: rerun the same
# stage after an interruption and verified outputs are skipped.
#
#   scripts/run_study1.sh train-r3   # ~30 min, third training replicate (seed 43)
#   scripts/run_study1.sh main       # ~7 h, 42 prompts x 4 seeds x (base + r1, r2, r3)
#   scripts/run_study1.sh sweep      # ~3 h, scale and checkpoint sweeps on 12 prompts
#   scripts/run_study1.sh metrics    # ~30 min, score everything (needs .venv-eval)
#
# Measured on one RTX 4090 (24 GB) under WSL2 with 47 GB RAM: 8-bit base images
# take ~27 s, images with an unfused adapter ~42 s.
set -euo pipefail
cd "$(dirname "$0")/.."

GEN=(.venv/bin/python -m evaluation.generate_pairs --quantization 8bit --resume)
OUT=evaluation/results/study1_8bit
R1=output/study1_pilot/study1_pilot.safetensors
R2=output/study1_pilot_v2/study1_pilot_v2.safetensors
R3=output/study1_pilot_r3/study1_pilot_r3.safetensors

case "${1:-}" in
  train-r3)
    if [ -f "$R3" ]; then echo "$R3 already exists; refusing to retrain over it"; exit 1; fi
    # Offline W&B: the run stays local and can be synced later with `wandb sync`.
    WANDB_MODE=offline ../ai-toolkit/.venv/bin/python ../ai-toolkit/run.py configs/study1_pilot_r3.yaml
    ;;
  main)
    # --resume on an empty directory simply starts a fresh run.
    "${GEN[@]}" --prompts evaluation/prompts_holdout.jsonl --seeds 1001 1002 1003 1004 \
      --output-dir "$OUT/main" --arm "r1=$R1" --arm "r2=$R2" --arm "r3=$R3"
    ;;
  sweep)
    ARMS=()
    for scale in 0 0.25 0.5 0.75 1.0 1.5; do ARMS+=(--arm "s$scale=$R1@$scale"); done
    for step in 100 200 300 400; do
      ARMS+=(--arm "c$step=output/study1_pilot/study1_pilot_$(printf '%09d' "$step").safetensors")
    done
    "${GEN[@]}" --prompts evaluation/prompts_holdout_sweep.jsonl --seeds 1001 1002 \
      --output-dir "$OUT/sweep" "${ARMS[@]}"
    ;;
  metrics)
    M=(.venv-eval/bin/python -m evaluation.metrics.run_metrics)
    "${M[@]}" --grids "$OUT/main/generation_manifest.jsonl" --output-dir results/metrics/main
    "${M[@]}" --grids "$OUT/sweep/generation_manifest.jsonl" --output-dir results/metrics/sweep
    "${M[@]}" --grids evaluation/results/holdout/generation_manifest.jsonl \
      --output-dir results/metrics/holdout_bf16
    ;;
  *)
    sed -n '2,12p' "$0"
    exit 2
    ;;
esac
