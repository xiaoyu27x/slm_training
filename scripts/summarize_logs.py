import math, re
from pathlib import Path

pattern = re.compile(r"step\s+(\d+):\s+train loss\s+([0-9.]+),\s+val loss\s+([0-9.]+)")

print(f"{'Model':<8} {'Step':>8} {'Train Loss':>12} {'Val Loss':>12} {'PPL':>12}")
print("-" * 58)

for name, path in [
    ("10m", Path("train_10m.log")),
    ("30m", Path("train_30m.log")),
    ("8h", Path("train_8h.log")),
]:
    if not path.exists():
        print(f"{name:<8} {'MISSING':>8}")
        continue
    matches = pattern.findall(path.read_text(errors="ignore"))
    if not matches:
        print(f"{name:<8} {'NO EVAL':>8}")
        continue
    step, train_loss, val_loss = matches[-1]
    train_loss, val_loss = float(train_loss), float(val_loss)
    print(f"{name:<8} {int(step):>8} {train_loss:>12.4f} {val_loss:>12.4f} {math.exp(val_loss):>12.2f}")
