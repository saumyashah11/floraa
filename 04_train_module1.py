"""
FLORA :: script 04 -- train the attribute-bottleneck network. This IS module 1.

Resumes automatically from the last Drive checkpoint, so a dropped Colab
session costs you at most one epoch.

    !python 04_train_module1.py --backbone swint --mode joint
    !python 04_train_module1.py --backbone resnet50 --mode baseline
    !python 04_train_module1.py --backbone dinov2s --mode joint --freeze_backbone
"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (MANIFESTS, CKPTS, RESULTS, IMG_SIZE, BATCH_SIZE, ACCUM_STEPS,
                    NUM_WORKERS, EPOCHS, LR_BACKBONE, LR_HEAD, WEIGHT_DECAY,
                    WARMUP_EPOCHS, LABEL_SMOOTH, PATIENCE, SEED, ZSL_UNSEEN)
from flora_model import (FloraDataset, FloraAttributeNet, flora_loss,
                         CAT_HEADS, severity_expectation)


def loaders(a):
    df = pd.read_csv(MANIFESTS / "manifest_attr.csv")
    seen = df[df.zsl_unseen == 0]
    classes = sorted(seen.canonical_label.unique())

    tr = seen[seen.split == "train"]
    va = seen[seen.split == "val"]
    print(f"train {len(tr):,} | val {len(va):,} | {len(classes)} seen classes")
    print(f"held out for module 2: {ZSL_UNSEEN}")

    ds_tr = FloraDataset(tr, classes, train=True, size=a.img_size)
    ds_va = FloraDataset(va, classes, train=False, size=a.img_size)

    # long-tailed data: sample inversely to class frequency
    freq = tr.canonical_label.value_counts()
    w = tr.canonical_label.map(lambda c: 1.0 / freq[c]).values
    sampler = WeightedRandomSampler(torch.DoubleTensor(w), len(w), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=a.batch_size, sampler=sampler,
                       num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                       persistent_workers=NUM_WORKERS > 0)
    dl_va = DataLoader(ds_va, batch_size=a.batch_size * 2, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)
    return dl_tr, dl_va, classes


@torch.no_grad()
def evaluate(model, dl, dev, classes):
    model.eval()
    preds, gts = defaultdict(list), defaultdict(list)
    sev_hat, sev_true = [], []
    for x, y in tqdm(dl, desc="val", leave=False):
        x = x.to(dev, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model(x)
        for h in CAT_HEADS:
            preds[h] += out[h].argmax(1).cpu().tolist()
            gts[h] += y[h].tolist()
        sev_hat += severity_expectation(out["severity"].float()).cpu().tolist()
        sev_true += y["severity"].tolist()
        preds["is_healthy"] += (torch.sigmoid(out["is_healthy"]) > 0.5).long().cpu().tolist()
        gts["is_healthy"] += y["is_healthy"].long().tolist()
        for k, tag in (("disease_bottleneck", "d_bneck"), ("disease_direct", "d_direct")):
            if k in out:
                preds[tag] += out[k].argmax(1).cpu().tolist()
                gts[tag] += y["disease"].tolist()

    m = {}
    for h in list(CAT_HEADS) + ["is_healthy", "d_bneck", "d_direct"]:
        if not preds[h]:
            continue
        p, g = np.array(preds[h]), np.array(gts[h])
        keep = g != -100
        m[f"{h}_acc"] = float((p[keep] == g[keep]).mean())
        m[f"{h}_f1"] = float(f1_score(g[keep], p[keep], average="macro", zero_division=0))
    m["severity_mae"] = float(np.abs(np.array(sev_hat) - np.array(sev_true)).mean())
    m["attr_mean_f1"] = float(np.mean([m[f"{h}_f1"] for h in CAT_HEADS]))
    return m


def main(a):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl_tr, dl_va, classes = loaders(a)

    model = FloraAttributeNet(a.backbone, n_disease=len(classes), mode=a.mode,
                              freeze_backbone=a.freeze_backbone,
                              img_size=a.img_size).to(dev)

    pre = CKPTS / f"pretrain_{a.backbone}.pt"
    if pre.exists() and not a.no_pretrain:
        sd = torch.load(pre, map_location="cpu")["state_dict"]
        missing, unexp = model.backbone.load_state_dict(sd, strict=False)
        print(f"loaded PlantVillage backbone ({len(missing)} missing, {len(unexp)} unexpected)")
    else:
        print("no PlantVillage checkpoint -- starting from ImageNet weights")

    head_p = [p for n, p in model.named_parameters()
              if not n.startswith("backbone.") and p.requires_grad]
    back_p = [p for n, p in model.named_parameters()
              if n.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": back_p, "lr": a.lr_backbone},
         {"params": head_p, "lr": a.lr_head}], weight_decay=WEIGHT_DECAY)

    spe = len(dl_tr) // ACCUM_STEPS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr_backbone, a.lr_head], total_steps=a.epochs * spe,
        pct_start=WARMUP_EPOCHS / max(a.epochs, 1))
    scaler = torch.cuda.amp.GradScaler()

    tag = f"m1_{a.backbone}_{a.mode}" + ("_frozen" if a.freeze_backbone else "")
    ck = CKPTS / f"{tag}.pt"
    start, best, bad, hist = 0, -1.0, 0, []
    if ck.exists() and not a.restart:
        s = torch.load(ck, map_location="cpu")
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
        start, best, hist = s["epoch"], s["best"], s["hist"]
        print(f"resumed {tag} at epoch {start}, best attr_mean_f1 {best:.4f}")

    for ep in range(start, a.epochs):
        model.train(); t0 = time.time(); agg = defaultdict(float); n = 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tqdm(dl_tr, desc=f"ep {ep+1}/{a.epochs}")):
            x = x.to(dev, non_blocking=True)
            y = {k: v.to(dev, non_blocking=True) for k, v in y.items()}
            with torch.cuda.amp.autocast():
                loss, parts = flora_loss(model(x), y, LABEL_SMOOTH)
                loss = loss / ACCUM_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
            for k, v in parts.items():
                agg[k] += v
            n += 1

        m = evaluate(model, dl_va, dev, classes)
        m["epoch"] = ep + 1
        m["train_loss"] = sum(agg.values()) / max(n, 1)
        m["minutes"] = round((time.time() - t0) / 60, 1)
        hist.append(m)
        print(f"ep {ep+1}: loss {m['train_loss']:.3f} | attrF1 {m['attr_mean_f1']:.4f} "
              f"| sevMAE {m['severity_mae']:.3f} "
              f"| bneck {m.get('d_bneck_acc', float('nan')):.4f} "
              f"| direct {m.get('d_direct_acc', float('nan')):.4f} "
              f"| {m['minutes']} min")

        score = m["attr_mean_f1"] + m.get("d_bneck_acc", m.get("d_direct_acc", 0))
        improved = score > best
        if improved:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "classes": classes,
                        "backbone": a.backbone, "mode": a.mode,
                        "img_size": a.img_size, "metrics": m}, CKPTS / f"{tag}_best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep + 1, "best": best, "hist": hist,
                    "classes": classes}, ck)
        (RESULTS / f"{tag}_history.json").write_text(json.dumps(hist, indent=2))
        if bad >= PATIENCE:
            print(f"early stop at epoch {ep+1}")
            break

    print(f"\ndone. best composite {best:.4f}. weights at {CKPTS / (tag + '_best.pt')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="swint")
    ap.add_argument("--mode", default="joint", choices=["baseline", "bottleneck", "joint"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--lr_backbone", type=float, default=LR_BACKBONE)
    ap.add_argument("--lr_head", type=float, default=LR_HEAD)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--no_pretrain", action="store_true")
    ap.add_argument("--restart", action="store_true")
    main(ap.parse_args())
