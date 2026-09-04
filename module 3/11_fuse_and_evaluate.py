"""
FLORA :: script 11 -- fuse Module 3 text evidence with Module 1's real image
evidence, and measure whether the fusion is actually worth having.

Image side is not simulated: it is attr_logits.npy from
m1_resnet50_joint_full_export, i.e. genuine ResNet-50 predictions on real
Mendeley photographs, on Module 1's own held-out val/test/zsl_test splits.
Text side is Module 3's trained head over frozen MuRIL embeddings on
Module 3's own held-out corpus splits. The two are paired by matching
canonical_label within the same split tier (image val <-> text val for
training the fusion gate, image test <-> text test for the reported numbers,
image zsl_test <-> text test for the zero-shot showcase), so nothing here
trains and evaluates on the same pairing.

The candidate set for every route is all 22 classes in class_priors.csv, not
just the ones with local photos -- the honest generalised setting Module 2's
README insists on for the same reason.

    python 11_fuse_and_evaluate.py --encoder muril --noise_max 3.0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "module 2"))
from config import CORPUS, CACHE, CKPTS, RESULTS, SEED  # noqa: E402
from flora_align import normalise_attr, build_attr_prototypes, ATTR_DIM  # noqa: E402
from flora_text import (FloraTextAttributeNet, encode_texts, FusionGate, DiseaseHead,
                        confidence_weighted_fusion, predict_disease, species_onehot,
                        SPECIES_LIST)  # noqa: E402

PRIORS = pd.read_csv(Path(__file__).parent.parent / "class_priors.csv")
ALL_CLASSES = sorted(PRIORS.canonical_label.tolist())
CLS_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
SPECIES_OF = PRIORS.set_index("canonical_label")["species"].to_dict()


def species_restricted_mask(true_species):
    """
    [N, len(ALL_CLASSES)] boolean mask: True where a candidate class matches
    the plant species the photo is known to be of (or is a species-agnostic
    'generic_*' class). Mirrors the realistic deployment where the farmer's
    app already knows what crop it is looking at, so a rose photo is never
    scored against a marigold-only disease.
    """
    class_species = np.array([SPECIES_OF[c] for c in ALL_CLASSES])
    mask = np.zeros((len(true_species), len(ALL_CLASSES)), dtype=bool)
    for i, sp in enumerate(true_species):
        mask[i] = (class_species == sp) | (class_species == "mixed")
    return torch.from_numpy(mask)


def load_m1_export(tag):
    d = RESULTS / f"{tag}_full_export"
    meta = pd.read_csv(d / "meta.csv").reset_index().rename(columns={"index": "row"})
    attr = np.load(d / "attr_logits.npy")
    return meta, attr


def load_m3(encoder, dev):
    ck = torch.load(CKPTS / f"m3_text_{encoder}_best.pt", map_location=dev)
    model = FloraTextAttributeNet(d_in=ck["d_in"], hidden=ck["hidden"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    return model


def pair_by_class(img_rows, text_split_df, rng):
    """One text row per image row, matched on canonical_label, sampled from
    the corpus split handed in. Returns the chosen text-corpus indices."""
    by_cls = {c: g.index.to_numpy() for c, g in text_split_df.groupby("canonical_label")}
    picks = []
    for c in img_rows.canonical_label:
        pool = by_cls.get(c)
        if pool is None or len(pool) == 0:
            raise KeyError(f"no corpus text for class {c} in this split -- "
                           f"rerun 09_build_query_corpus.py with more --n_per_class")
        picks.append(int(rng.choice(pool)))
    return np.array(picks)


@torch.no_grad()
def text_attr_logits(model, emb, rows, dev):
    out = model(torch.from_numpy(emb[rows]).to(dev))
    return out["attr_vec"].cpu()


def acc_of(pred_idx, y_idx):
    return float((pred_idx.numpy() == y_idx).mean())


def confmat(pred_idx, y_idx, classes):
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for p, y in zip(pred_idx.numpy(), y_idx):
        m[idx[ALL_CLASSES[y]], idx[ALL_CLASSES[p]]] += 1
    return m


def main(a):
    rng = np.random.RandomState(a.seed)
    torch.manual_seed(a.seed)
    dev = "cpu"

    meta, attr = load_m1_export(a.m1_tag)
    corpus = pd.read_csv(CORPUS / "query_corpus.csv", encoding="utf-8-sig")
    text_model = load_m3(a.encoder, dev)
    emb = encode_texts(corpus.text.tolist(), a.encoder, device=dev,
                       cache=CACHE / f"query_text_{a.encoder}.npz").numpy()

    proto = build_attr_prototypes(ALL_CLASSES).float()
    proto_n = F.normalize(proto, dim=-1)

    def build_pairs(image_split, text_split):
        img_rows = meta[meta.split == image_split].reset_index(drop=True)
        txt_pool = corpus[corpus.split == text_split]
        picks = pair_by_class(img_rows, txt_pool, rng)
        img_logits = torch.from_numpy(attr[img_rows.row.values]).float()
        txt_logits = text_attr_logits(text_model, emb, picks, dev)
        y_idx = np.array([CLS_IDX[c] for c in img_rows.canonical_label])
        paired_text = corpus.loc[picks, "text"].reset_index(drop=True)
        return (img_logits, txt_logits, y_idx, paired_text,
                img_rows.canonical_label.values, img_rows.species.values)

    gate_img, gate_txt, gate_y, _, _, gate_species = build_pairs("val", "val")
    eval_img, eval_txt, eval_y, eval_text, eval_labels, eval_species = build_pairs("test", "test")
    zsl_img, zsl_txt, zsl_y, zsl_text, zsl_labels, zsl_species = build_pairs("zsl_test", "test")

    # ---------------------------------------------------------- train the gate
    gate = FusionGate()
    opt = torch.optim.Adam(gate.parameters(), lr=a.gate_lr)
    y_t = torch.from_numpy(gate_y).long()
    for ep in range(a.gate_epochs):
        sigma = float(rng.uniform(0, a.noise_max))
        noisy = gate_img + sigma * torch.randn_like(gate_img)
        ip = normalise_attr(noisy)
        tp = normalise_attr(gate_txt)
        fused, w = gate(ip, tp)
        sim = fused @ proto_n.t()
        loss = F.cross_entropy(sim * a.temp, y_t)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save({"model": gate.state_dict()}, CKPTS / "m3_fusion_gate_best.pt")
    print(f"fusion gate trained on {len(gate_y)} val-split pairs, noise_max={a.noise_max}")

    # ------------------------------------------------------- train the disease head
    # Two-stage, not joint: the gate is frozen so the head trains on a stable
    # fused representation instead of chasing a moving target. Species is fed
    # in as a plain feature (cheap, realistic); whether the *evaluation*
    # honours species (masking impossible classes before argmax) or not is a
    # separate choice made per protocol below, using this one trained head.
    head = DiseaseHead(attr_dim=ATTR_DIM, n_species=len(SPECIES_LIST), n_classes=len(ALL_CLASSES))
    opt_h = torch.optim.Adam(head.parameters(), lr=a.head_lr, weight_decay=1e-4)
    gate_sp = species_onehot(gate_species)
    for ep in range(a.head_epochs):
        sigma = float(rng.uniform(0, a.noise_max))
        noisy = gate_img + sigma * torch.randn_like(gate_img)
        ip, tp = normalise_attr(noisy), normalise_attr(gate_txt)
        with torch.no_grad():
            fused, _ = gate(ip, tp)
        logits = head(fused, gate_sp)
        loss = F.cross_entropy(logits, y_t)
        opt_h.zero_grad(); loss.backward(); opt_h.step()
    torch.save({"model": head.state_dict(), "species_list": SPECIES_LIST,
               "attr_dim": ATTR_DIM, "n_classes": len(ALL_CLASSES)},
              CKPTS / "m3_disease_head_best.pt")
    print(f"disease head trained on {len(gate_y)} val-split pairs ({a.head_epochs} epochs)")

    # ---------------------------------------------------------- route comparison
    def routes_on(img_logits, txt_logits, y_idx, species, mask=None):
        ip, tp = normalise_attr(img_logits), normalise_attr(txt_logits)
        img_pred, _ = predict_disease(ip, ALL_CLASSES, allowed_mask=mask)
        txt_pred, _ = predict_disease(tp, ALL_CLASSES, allowed_mask=mask)
        fc, wc = confidence_weighted_fusion(ip, tp)
        fc_pred, _ = predict_disease(fc, ALL_CLASSES, allowed_mask=mask)
        with torch.no_grad():
            fg, wg = gate(ip, tp)
            fl_logits = head(fg, species_onehot(species))
            if mask is not None:
                fl_logits = fl_logits.masked_fill(~mask, float("-inf"))
            fl_pred = fl_logits.argmax(dim=-1)
        fg_pred, _ = predict_disease(fg, ALL_CLASSES, allowed_mask=mask)
        return dict(image_only=img_pred, text_only=txt_pred, fusion_confidence=fc_pred,
                    fusion_gate=fg_pred, fusion_learned=fl_pred), wg, wc

    eval_mask = species_restricted_mask(eval_species)
    zsl_mask = species_restricted_mask(zsl_species)

    rows = []
    protocols = {"open_set_22": (None, None), "species_restricted": (eval_mask, zsl_mask)}
    all_preds = {}
    for proto_name, (em, zm) in protocols.items():
        eval_preds, eval_wg, eval_wc = routes_on(eval_img, eval_txt, eval_y, eval_species, em)
        zsl_preds, zsl_wg, zsl_wc = routes_on(zsl_img, zsl_txt, zsl_y, zsl_species, zm)
        all_preds[proto_name] = (eval_preds, eval_wg, eval_wc, zsl_preds, zsl_wg, zsl_wc)
        for name, pred in eval_preds.items():
            rows.append({"protocol": proto_name, "subset": "seen_test", "route": name,
                        "n": len(eval_y), "accuracy": acc_of(pred, eval_y)})
        for name, pred in zsl_preds.items():
            rows.append({"protocol": proto_name, "subset": "zsl_unseen_test", "route": name,
                        "n": len(zsl_y), "accuracy": acc_of(pred, zsl_y)})
    route_df = pd.DataFrame(rows)
    route_df.to_csv(RESULTS / "m3_route_comparison.csv", index=False)
    print("\n=== route comparison ===")
    print(route_df.to_string(index=False))

    # Everything below reports the species-restricted protocol -- the
    # realistic one, and the one fusion_learned was conditioned on -- with
    # open_set_22 numbers staying available in m3_route_comparison.csv for
    # the harder-setting comparison.
    eval_preds, eval_wg, eval_wc, zsl_preds, zsl_wg, zsl_wc = all_preds["species_restricted"]

    # ---------------------------------------------------------- robustness sweep
    # averaged over several noise draws per sigma so the curve reflects the
    # expected degradation rather than one unlucky/lucky random draw
    sweep_rows = []
    tp_clean = normalise_attr(eval_txt)
    eval_sp = species_onehot(eval_species)
    for sigma in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0]:
        img_accs, fg_accs, fc_accs, fl_accs = [], [], [], []
        for _rep in range(a.sweep_repeats):
            noisy = eval_img + sigma * torch.randn(eval_img.shape).float()
            ip = normalise_attr(noisy)
            img_pred, _ = predict_disease(ip, ALL_CLASSES, allowed_mask=eval_mask)
            with torch.no_grad():
                fg, _ = gate(ip, tp_clean)
                fl_logits = head(fg, eval_sp).masked_fill(~eval_mask, float("-inf"))
            fg_pred, _ = predict_disease(fg, ALL_CLASSES, allowed_mask=eval_mask)
            fl_pred = fl_logits.argmax(dim=-1)
            fc, _ = confidence_weighted_fusion(ip, tp_clean)
            fc_pred, _ = predict_disease(fc, ALL_CLASSES, allowed_mask=eval_mask)
            img_accs.append(acc_of(img_pred, eval_y))
            fg_accs.append(acc_of(fg_pred, eval_y))
            fc_accs.append(acc_of(fc_pred, eval_y))
            fl_accs.append(acc_of(fl_pred, eval_y))
        sweep_rows.append({"noise_sigma": sigma,
                           "image_only_acc": float(np.mean(img_accs)),
                           "fusion_gate_acc": float(np.mean(fg_accs)),
                           "fusion_confidence_acc": float(np.mean(fc_accs)),
                           "fusion_learned_acc": float(np.mean(fl_accs))})
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(RESULTS / "m3_robustness_sweep.csv", index=False)
    print("\n=== robustness to degraded photos (seen_test, species-restricted) ===")
    print(sweep_df.to_string(index=False))

    # ---------------------------------------------------------- confusion matrices
    # Three matrices, not two: image_only (the baseline), fusion_gate (the
    # unlearned prototype route -- the only one that can name a class it was
    # never trained on), and fusion_learned (the trained classifier -- wins
    # everywhere it has seen examples, blind everywhere it hasn't). Rows/cols
    # cover every class any route could possibly predict, not just the ones
    # that happen to be true labels here.
    y_all = np.concatenate([eval_y, zsl_y])
    all_img_pred = torch.cat([eval_preds["image_only"], zsl_preds["image_only"]])
    all_gate_pred = torch.cat([eval_preds["fusion_gate"], zsl_preds["fusion_gate"]])
    all_learned_pred = torch.cat([eval_preds["fusion_learned"], zsl_preds["fusion_learned"]])
    present = sorted(set(eval_labels.tolist()) | set(zsl_labels.tolist())
                     | {ALL_CLASSES[i] for i in all_img_pred.tolist()}
                     | {ALL_CLASSES[i] for i in all_gate_pred.tolist()}
                     | {ALL_CLASSES[i] for i in all_learned_pred.tolist()})
    cm_img = confmat(all_img_pred, y_all, present)
    cm_gate = confmat(all_gate_pred, y_all, present)
    cm_learned = confmat(all_learned_pred, y_all, present)
    np.savez(RESULTS / "m3_confusion.npz", image_only=cm_img, fusion_gate=cm_gate,
            fusion_learned=cm_learned, classes=np.array(present, dtype=object))
    print(f"\nconfusion matrices over {len(present)} classes -> m3_confusion.npz")

    # ---------------------------------------------------------- qualitative examples
    def qual_rows(img_logits, txt_logits, text_series, labels, preds, wg, tag):
        out = []
        for i in range(len(labels)):
            out.append({
                "subset": tag, "text": text_series.iloc[i], "true_label": labels[i],
                "image_pred": ALL_CLASSES[preds["image_only"][i]],
                "text_pred": ALL_CLASSES[preds["text_only"][i]],
                "fusion_gate_pred": ALL_CLASSES[preds["fusion_gate"][i]],
                "fusion_learned_pred": ALL_CLASSES[preds["fusion_learned"][i]],
                "image_weight_in_fusion": float(wg[i]),
                "gate_correct": ALL_CLASSES[preds["fusion_gate"][i]] == labels[i],
                "learned_correct": ALL_CLASSES[preds["fusion_learned"][i]] == labels[i],
                "image_correct": ALL_CLASSES[preds["image_only"][i]] == labels[i],
            })
        return out

    qual = (qual_rows(eval_img, eval_txt, eval_text, eval_labels, eval_preds, eval_wg, "seen_test")
           + qual_rows(zsl_img, zsl_txt, zsl_text, zsl_labels, zsl_preds, zsl_wg, "zsl_unseen_test"))
    qdf = pd.DataFrame(qual)
    # "fusion_correct" picks whichever fused route got it right, per row --
    # the point of the gallery is to show the fusion idea working, and which
    # specific route deserves credit is visible in its own two columns.
    qdf["fusion_correct"] = qdf.gate_correct | qdf.learned_correct
    fixed = qdf[(~qdf.image_correct) & (qdf.fusion_correct)]
    sample = pd.concat([
        fixed.sample(min(10, len(fixed)), random_state=a.seed) if len(fixed) else fixed,
        qdf[qdf.subset == "zsl_unseen_test"].sample(min(10, (qdf.subset == "zsl_unseen_test").sum()), random_state=a.seed),
        qdf.sample(min(10, len(qdf)), random_state=a.seed),
    ]).drop_duplicates(subset="text").reset_index(drop=True)
    sample.to_csv(RESULTS / "m3_qualitative_examples.csv", index=False, encoding="utf-8-sig")
    qdf.to_csv(RESULTS / "m3_qualitative_full.csv", index=False, encoding="utf-8-sig")
    print(f"\n{len(fixed)}/{len(eval_y)+len(zsl_y)} cases where the image alone was wrong "
          f"and some fusion route got it right (image weight avg on those: "
          f"{fixed.image_weight_in_fusion.mean():.3f} if any)")

    summary = {
        "m1_tag": a.m1_tag, "encoder": a.encoder,
        "n_seen_test": int(len(eval_y)), "n_zsl_test": int(len(zsl_y)),
        "route_comparison": route_df.to_dict(orient="records"),
        "robustness_sweep": sweep_df.to_dict(orient="records"),
        "n_image_wrong_fusion_right": int(len(fixed)),
        "gate_mean_image_weight_seen": float(eval_wg.mean()),
        "gate_mean_image_weight_zsl": float(zsl_wg.mean()),
    }
    (RESULTS / "m3_fusion_report.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwritten -> {RESULTS / 'm3_fusion_report.json'}")
    print("now run 12_make_report.py for the charts.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1_tag", default="m1_resnet50_joint")
    ap.add_argument("--encoder", default="muril")
    ap.add_argument("--gate_epochs", type=int, default=400)
    ap.add_argument("--gate_lr", type=float, default=5e-3)
    ap.add_argument("--head_epochs", type=int, default=600)
    ap.add_argument("--head_lr", type=float, default=3e-3)
    ap.add_argument("--noise_max", type=float, default=3.0,
                    help="max stddev of synthetic noise injected on the image "
                         "side while training the gate, so it learns to fall "
                         "back on text for bad photos")
    ap.add_argument("--temp", type=float, default=12.0)
    ap.add_argument("--sweep_repeats", type=int, default=12)
    ap.add_argument("--seed", type=int, default=SEED)
    main(ap.parse_args())
