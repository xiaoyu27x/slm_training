#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[1/2] 30-minute experiment"
python -u train.py config/train_gpt2_v100.py \
  --dataset=owt_60k \
  --out_dir=out-owt-30m \
  --max_iters=1600 \
  --lr_decay_iters=1600 \
  --warmup_iters=80 \
  --eval_interval=1600 \
  2>&1 | tee train_30m.log

echo "[2/2] 8-hour experiment"
python -u train.py config/train_gpt2_v100.py \
  --dataset=owt_1m \
  --out_dir=out-owt-8h \
  --max_iters=25850 \
  --lr_decay_iters=25850 \
  --warmup_iters=1293 \
  --eval_interval=25850 \
  2>&1 | tee train_8h.log
