#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

python -u train.py config/train_gpt2_v100.py \
  --dataset=owt_20k \
  --out_dir=out-owt-10m \
  --max_iters=520 \
  --lr_decay_iters=520 \
  --warmup_iters=26 \
  --eval_interval=520 \
  2>&1 | tee train_10m.log
