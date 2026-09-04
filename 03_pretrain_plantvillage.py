"""
FLORA :: script 03 -- domain adaptation of the backbone on PlantVillage.

Run this ONCE per backbone. It writes a backbone-only state dict to Drive which
script 04 loads. Roughly 12-18 minutes per epoch on a free T4, so 4 epochs is
about one session. Do not repeat it.

    !python 03_pretrain_plantvillage.py --backbone swint --epochs 4
"""
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW, CKPTS, BACKBONES, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, PRETRAIN_SET
from flora_model import build_transforms

import timm


def find_split(root: Path, want: str):
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower().startswith(want):
            if any(c.is_dir() for c in p.iterdir()):
                return p
    return None


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = RAW / PRETRAIN_SET[1]
    if not root.exists():
        raise SystemExit(
            f"{root} missing. Download it first:\n"
            f"  !kaggle datasets download -d {PRETRAIN_SET[0]} -p {root} --unzip")

    tr_dir = find_split(root, "train")
    va_dir = find_split(root, "valid") or find_split(root, "val")
    if tr_dir is None:
        raise SystemExit(f"Could not locate a train/ folder under {root}")
    print("train:", tr_dir, "\nvalid:", va_dir)

    tr = ImageFolder(tr_dir, build_transforms(True))
    va = ImageFolder(va_dir, build_transforms(False)) if va_dir else None
    print(f"{len(tr):,} train images, {len(tr.classes)} classes")

    dl_tr = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                       persistent_workers=NUM_WORKERS > 0)
    dl_va = (DataLoader(va, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS,
                        pin_memory=True) if va else None)

    name = BACKBONES.get(a.backbone, a.backbone)
    kw = dict(pretrained=True, num_classes=len(tr.classes))
    if "dinov2" in name:
        kw.update(img_size=IMG_SIZE, dynamic_img_size=True)
    model = timm.create_model(name, **kw).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)
    steps = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=steps, pct_start=0.15)
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    ck = CKPTS / f"pretrain_{a.backbone}.pt"
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for i, (x, y) in enumerate(tqdm(dl_tr, desc=f"ep {ep+1}/{a.epochs}")):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.cuda.amp.autocast():
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run += float(loss)
            if a.max_steps and i >= a.max_steps:
                break

        acc = float("nan")
        if dl_va:
            model.eval(); c = n = 0
            with torch.no_grad(), torch.cuda.amp.autocast():
                for j, (x, y) in enumerate(dl_va):
                    p = model(x.to(dev)).argmax(1).cpu()
                    c += (p == y).sum().item(); n += len(y)
                    if j >= 60:
                        break
            acc = c / max(n, 1)
        print(f"epoch {ep+1}: loss {run/max(i+1,1):.4f}  val~{acc:.4f}  "
              f"{(time.time()-t0)/60:.1f} min")

        # backbone weights only -- the 38-class head is thrown away
        sd = {k: v for k, v in model.state_dict().items()
              if not k.startswith(("head.", "fc.", "classifier."))}
        torch.save({"backbone": a.backbone, "state_dict": sd,
                    "epoch": ep + 1, "n_classes": len(tr.classes)}, ck)
        print(f"  saved {ck}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="swint", choices=list(BACKBONES))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="cap steps per epoch if your session is short")
    main(ap.parse_args())
