#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p results/generated

run_case () {
  model="$1"
  out="$2"
  prompt="$3"
  file="$4"

  python sample.py \
    --out_dir="$out" \
    --start="$prompt" \
    --num_samples=1 \
    --max_new_tokens=100 \
    --temperature=1.0 \
    --top_k=1 \
    2>&1 | tee "results/generated/$file"
}

for model in 10m 30m 8h; do
  case "$model" in
    10m) out="out-owt-10m" ;;
    30m) out="out-owt-30m" ;;
    8h) out="out-owt-8h" ;;
  esac

  run_case "$model" "$out" "Paris is the capital of" "title1_${model}.txt"
  run_case "$model" "$out" "A computer processor is responsible for" "title2_${model}.txt"
  run_case "$model" "$out" "Once upon a time there was a young girl who" "title3_${model}.txt"
done
