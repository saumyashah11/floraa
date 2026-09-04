"""
FLORA :: script 10 -- train Module 3's text attribute heads.

Operates on frozen MuRIL/IndicBERT sentence embeddings, cached to disk after
the first pass -- exactly the pattern 07_train_alignment.py uses for the
monograph text tower. A full run is under a minute on CPU once the corpus is
embedded once. Fine-tuning the encoder end to end is a GPU/Colab job (pass
--finetune_backbone); on ~4.5k short sentences without it, MuRIL memorises
the corpus in a handful of steps and the held-out numbers stop meaning
anything, the same trap Module 2's README warns about for its own text tower.

    python 10_train_module3.py --encoder muril --epochs 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORPUS, CACHE, CKPTS, RESULTS, SEED  # noqa: E402
from flora_model import CAT_HEADS, CAT_VALUES            # noqa: E402
from flora_text import FloraTextAttributeNet, encode_texts, text_loss, TEXT_ENCODERS  # noqa: E402


def pack(df, emb, cat_to_idx):
    rows = df.index.to_numpy()
    y = {h: torch.tensor([cat_to_idx[h][v] for v in df[h]], dtype=torch.long) for h in CAT_HEADS}
    y["severity"] = torch.tensor(df.severity.values, dtype=torch.long)
    y["coverage"] = torch.tensor(df.coverage.values, dtype=torch.float32)
    y["is_healthy"] = torch.tensor(df.is_healthy.values, dtype=torch.float32)
    keys = list(y.keys())
    return TensorDataset(torch.from_numpy(emb[rows]), *[y[k] for k in keys]), keys


@torch.no_grad()
def evaluate(model, dl, keys, device):
    model.eval()
    correct = {h: 0 for h in CAT_HEADS + ["severity", "is_healthy"]}
    abs_err_cov, n = 0.0, 0
    for batch in dl:
        x = batch[0].to(device)
        y = dict(zip(keys, [b.to(device) for b in batch[1:]]))
        out = model(x)
        for h in CAT_HEADS + ["severity"]:
            correct[h] += int((out[h].argmax(1) == y[h]).sum())
        correct["is_healthy"] += int(((torch.sigmoid(out["is_healthy"]) > 0.5).float() == y["is_healthy"]).sum())
        abs_err_cov += float((torch.sigmoid(out["coverage"]) - y["coverage"]).abs().sum())
        n += x.shape[0]
    acc = {h: correct[h] / n for h in correct}
    acc["coverage_mae"] = abs_err_cov / n
    return acc


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(CORPUS / "query_corpus.csv", encoding="utf-8-sig")
    cat_to_idx = {h: {v: i for i, v in enumerate(CAT_VALUES[h])} for h in CAT_HEADS}

    cache = CACHE / f"query_text_{a.encoder}.npz"
    emb = encode_texts(df.text.tolist(), a.encoder, device=dev, cache=cache).numpy()
    print(f"embedded {len(df):,} queries with {a.encoder} -> {emb.shape}")

    tr_ds, keys = pack(df[df.split == "train"], emb, cat_to_idx)
    va_ds, _ = pack(df[df.split == "val"], emb, cat_to_idx)
    te_ds, _ = pack(df[df.split == "test"], emb, cat_to_idx)
    dl_tr = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=True)
    dl_va = DataLoader(va_ds, batch_size=512)
    dl_te = DataLoader(te_ds, batch_size=512)

    model = FloraTextAttributeNet(d_in=emb.shape[1], hidden=a.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * max(len(dl_tr), 1))

    tag = f"m3_text_{a.encoder}"
    best_score, hist = -1.0, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); agg, n = 0.0, 0
        for batch in dl_tr:
            x = batch[0].to(dev)
            y = dict(zip(keys, [b.to(dev) for b in batch[1:]]))
            out = model(x)
            loss, _ = text_loss(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            agg += float(loss); n += 1

        va = evaluate(model, dl_va, keys, dev)
        score = np.mean([va[h] for h in CAT_HEADS + ["severity", "is_healthy"]])
        hist.append({"epoch": ep + 1, "loss": agg / max(n, 1), "val": va, "val_mean_acc": float(score)})
        if (ep + 1) % max(1, a.epochs // 15) == 0 or ep == 0:
            print(f"ep {ep+1:>3}: loss {agg/max(n,1):.4f}  val_mean_acc {score:.4f}  "
                  f"coverage_mae {va['coverage_mae']:.4f}  ({time.time()-t0:.1f}s)")
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "d_in": emb.shape[1],
                        "hidden": a.hidden, "encoder": a.encoder},
                       CKPTS / f"{tag}_best.pt")

    ck = torch.load(CKPTS / f"{tag}_best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    te = evaluate(model, dl_te, keys, dev)
    print("\n=== held-out test-split accuracy (text-only, per attribute head) ===")
    for h in CAT_HEADS + ["severity", "is_healthy"]:
        print(f"  {h:<14} {te[h]:.4f}")
    print(f"  {'coverage_mae':<14} {te['coverage_mae']:.4f}")

    (RESULTS / f"{tag}_history.json").write_text(json.dumps(hist, indent=2))
    (RESULTS / f"{tag}_test_report.json").write_text(json.dumps(te, indent=2))
    print(f"\nbest val_mean_acc {best_score:.4f} -> {CKPTS / (tag + '_best.pt')}")
    print("now run 11_fuse_and_evaluate.py to fuse this with Module 1's real image evidence.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="muril", choices=list(TEXT_ENCODERS.keys()))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=SEED)
    main(ap.parse_args())
