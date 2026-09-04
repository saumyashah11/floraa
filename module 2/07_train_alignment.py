"""
FLORA :: script 07 -- train the vision-language alignment.

Operates entirely on the vectors dumped by script 06, so there is no image
decoding and no backbone forward pass. A full run takes two to four minutes on
a T4 and works on CPU if the GPU is busy. Train the seen classes only; the
three zero-shot classes must never enter this loop.

    !python 07_train_alignment.py --tag m1_swint_joint --encoder mpnet
    !python 07_train_alignment.py --tag m1_swint_joint --use_features   # ablation
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS, CKPTS, CACHE, SEED, ZSL_UNSEEN
from flora_align import (FloraAligner, load_monographs, encode_texts,
                         build_attr_prototypes, normalise_attr, alignment_loss)


def load_export(tag):
    d = RESULTS / f"{tag}_full_export"
    meta = pd.read_csv(d / "meta.csv")
    attr = np.load(d / "attr_logits.npy")
    feat = np.load(d / "features.npy")
    assert len(meta) == len(attr) == len(feat), "export length mismatch"
    return meta, attr, feat


def main(a):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta, attr, feat = load_export(a.tag)

    all_classes = sorted(meta.canonical_label.unique())
    seen = [c for c in all_classes if c not in ZSL_UNSEEN]
    unseen = [c for c in all_classes if c in ZSL_UNSEEN]
    if not unseen:
        print("WARNING: no zero-shot classes present. Check ZSL_UNSEEN in config.py.")
    print(f"{len(seen)} seen | {len(unseen)} unseen -> {unseen}")

    # text bank for EVERY class, seen and unseen alike
    texts = load_monographs(all_classes)
    cache = CACHE / f"text_{a.encoder}.npz"
    tb, tclasses = encode_texts(texts, a.encoder, device=dev, cache=cache)
    tb = tb.to(dev)                                   # [C, V, D]
    idx_of = {c: i for i, c in enumerate(tclasses)}
    seen_idx = torch.tensor([idx_of[c] for c in seen], device=dev)
    print(f"text bank {tuple(tb.shape)} from {a.encoder}")

    proto = build_attr_prototypes(tclasses).to(dev)
    proto_n = F.normalize(proto, dim=-1)

    # training rows: seen classes, train split only
    m = meta.reset_index().rename(columns={"index": "row"})
    tr = m[(m.split == "train") & (~m.canonical_label.isin(ZSL_UNSEEN))]
    va = m[(m.split == "val") & (~m.canonical_label.isin(ZSL_UNSEEN))]
    print(f"train {len(tr):,} | val {len(va):,}")

    def pack(sub):
        y = torch.tensor([seen.index(c) for c in sub.canonical_label], dtype=torch.long)
        return TensorDataset(torch.from_numpy(attr[sub.row.values]),
                             torch.from_numpy(feat[sub.row.values]), y)

    dl_tr = DataLoader(pack(tr), batch_size=a.batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(pack(va), batch_size=512, shuffle=False)

    model = FloraAligner(d_text=tb.shape[-1], d_shared=a.d_shared,
                         use_features=a.use_features,
                         d_feat=feat.shape[1], temp=a.temp).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * max(len(dl_tr), 1))

    tag = f"m2_{a.tag}_{a.encoder}" + ("_feat" if a.use_features else "")
    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); agg, n = 0.0, 0
        for x, f, y in dl_tr:
            x, f, y = x.to(dev), f.to(dev), y.to(dev)
            # sample one monograph variant per class per step
            v = torch.randint(0, tb.shape[1], (1,)).item()
            bank = tb[seen_idx, v, :]
            logits = model(x, bank, f)
            ps = normalise_attr(x) @ proto_n[seen_idx].t()
            loss, _ = alignment_loss(logits, y, ps, a.w_proto)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            agg += float(loss); n += 1

        model.eval(); c = t = 0
        with torch.no_grad():
            bank = tb[seen_idx].mean(1)          # average the variants at eval
            for x, f, y in dl_va:
                p = model(x.to(dev), bank, f.to(dev)).argmax(1).cpu()
                c += int((p == y).sum()); t += len(y)
        acc = c / max(t, 1)
        hist.append({"epoch": ep + 1, "loss": agg / max(n, 1), "val_seen_acc": acc})
        print(f"ep {ep+1:>3}: loss {agg/max(n,1):.4f}  val seen acc {acc:.4f}  "
              f"({time.time()-t0:.0f}s)")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "seen": seen,
                        "unseen": unseen, "all_classes": tclasses,
                        "encoder": a.encoder, "use_features": a.use_features,
                        "d_shared": a.d_shared, "d_text": tb.shape[-1],
                        "d_feat": feat.shape[1], "m1_tag": a.tag},
                       CKPTS / f"{tag}_best.pt")

    (RESULTS / f"{tag}_history.json").write_text(json.dumps(hist, indent=2))
    print(f"\nbest seen-class val accuracy {best:.4f} -> {CKPTS / (tag + '_best.pt')}")
    print("now run 08_zsl_eval.py to get the number that actually matters.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m1_swint_joint")
    ap.add_argument("--encoder", default="mpnet",
                    choices=["mpnet", "bge", "minilm", "scibert"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_shared", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--w_proto", type=float, default=0.3)
    ap.add_argument("--use_features", action="store_true",
                    help="ABLATION ONLY: leaks past the bottleneck")
    main(ap.parse_args())
