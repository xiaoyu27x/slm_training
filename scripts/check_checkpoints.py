from pathlib import Path
import torch

for path in [
    Path("out-owt-10m/ckpt.pt"),
    Path("out-owt-30m/ckpt.pt"),
    Path("out-owt-8h/ckpt.pt"),
]:
    print("=" * 70)
    print(path)
    if not path.exists():
        print("MISSING")
        continue
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    print("iter_num      :", ckpt.get("iter_num"))
    print("best_val_loss :", ckpt.get("best_val_loss"))
    print("dataset       :", cfg.get("dataset"))
    print("max_iters     :", cfg.get("max_iters"))
    print("out_dir       :", cfg.get("out_dir"))
