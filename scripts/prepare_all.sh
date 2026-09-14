#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

python data/owt_20k/prepare.py
python data/owt_60k/prepare.py
python data/owt_1m/prepare.py
