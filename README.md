# nanoGPT OpenWebText Scaling Experiments on a Single Tesla V100 32GB

A reproducible experiment wrapper for [nanoGPT](https://github.com/karpathy/nanoGPT).

## Final experiment series

| Experiment | OpenWebText subset | max_iters |
|---|---:|---:|
| 10 min | 20,000 docs | 520 |
| 30 min | 60,000 docs | 1,600 |
| 8 h | 1,000,000 docs | 25,850 |

## Model / hardware

- GPT-2 124M style architecture
- 12 layers, 12 heads, 768 embedding
- context length 1024
- GPT-2 BPE tokenizer
- FP16
- NVIDIA Tesla V100-PCIE-32GB

Environment:

```text
Ubuntu 22.04
Python 3.10
PyTorch 2.1.2+cu118
NumPy 1.26.4
Tesla V100-PCIE-32GB
```

Use a project `.venv`.

## Upstream nanoGPT code

Keep:

```text
model.py
train.py
sample.py
```

Adds:

```text
config/train_gpt2_v100.py
data/owt_20k/prepare.py
data/owt_60k/prepare.py
data/owt_1m/prepare.py
scripts/*
```

## Setup

Clone nanoGPT first:

```bash
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
```

Copy this package into the nanoGPT root.

Create the isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-v100.txt
```

Verify:

```bash
python - <<'PY'
import torch, numpy
print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CC:", torch.cuda.get_device_capability(0))
PY
```

## Prepare data

```bash
bash scripts/prepare_all.sh
```

Expected binary files:

```text
data/owt_20k/train.bin
data/owt_20k/val.bin
data/owt_60k/train.bin
data/owt_60k/val.bin
data/owt_1m/train.bin
data/owt_1m/val.bin
```

Validated 20K preparation:

```text
Train documents : 19,800
Val documents   : 200
Train tokens    : 22,357,004
Val tokens      : 205,082
Total tokens    : 22,562,086
```

## Train

10-minute model:

```bash
tmux new -s owt10m
bash scripts/train_10m.sh
```

30-minute + 8-hour automatic sequence:

```bash
tmux new -s nanogpt_30m_8h
bash scripts/train_30m_8h.sh
```

Expected checkpoints:

```text
out-owt-10m/ckpt.pt
out-owt-30m/ckpt.pt
out-owt-8h/ckpt.pt
```

Each experiment starts from scratch.

## Monitor

```bash
bash scripts/monitor.sh train_10m.log 520
bash scripts/monitor.sh train_30m.log 1600
bash scripts/monitor.sh train_8h.log 25850
```

## Test all 3 models x 3 prompts = 9 runs

```bash
bash scripts/test_9_runs.sh
```

Prompts:

```text
Paris is the capital of
A computer processor is responsible for
Once upon a time there was a young girl who
```

The test script uses deterministic decoding (`top_k=1`) for fairer comparison.

## Summarize quantitative results

```bash
python scripts/summarize_logs.py
python scripts/check_checkpoints.py
```

## Observed behavior

- 10 min: basic English structure, weak semantics and coherence.
- 30 min: more article-like paragraphs and better local structure; factuality remains poor.
- 8 h: stronger fluency, punctuation and local coherence; factual recall is still unreliable.

These are small GPT-2-style models trained from scratch, not instruction-tuned chat models.

## Do not upload large artifacts to GitHub

`.gitignore` excludes:

- `train.bin`
- `val.bin`
- `ckpt.pt`
- `.venv/`
- logs
- cache files

See `UPSTREAM_NOTICE.md` for nanoGPT attribution.
