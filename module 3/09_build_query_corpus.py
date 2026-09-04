"""
FLORA :: script 09 -- build the labelled farmer-query corpus.

Templates x query_vocabulary.yaml x class_priors.csv. Categorical symptom
labels (colour, margin, distribution, texture) come straight from the class
prior, exactly as 02_build_attribute_labels.py does for images: symptom
description is a property of the disease, not of one farmer's phone camera.
Severity and coverage are perturbed per row, and the phrase chosen to
describe them tracks that perturbation, so text and regression target stay
consistent the way one real complaint would be.

    python 09_build_query_corpus.py --n_per_class 220
"""
import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORPUS, SEED  # noqa: E402

HERE = Path(__file__).parent
VOCAB = yaml.safe_load((HERE / "query_vocabulary.yaml").read_text(encoding="utf-8"))
PRIORS = pd.read_csv(HERE.parent / "class_priors.csv")

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def script_mix(text):
    has_deva, has_latin = bool(DEVANAGARI.search(text)), bool(LATIN.search(text))
    if has_deva and has_latin:
        return "mixed_script"
    if has_deva:
        return "devanagari"
    return "latin_hinglish"


def nearest_coverage_phrase(target):
    best = min(VOCAB["coverage_phrases"], key=lambda e: abs(e["value"] - target))
    return best["phrase"], best["value"]


def apply_typos(text, rng, p=0.35):
    if rng.random() > p:
        return text
    a, b = rng.choice(VOCAB["typo_swaps"])
    return text.replace(a, b) if a in text else text


def build_sentence(rng, row):
    parts = []
    opener = rng.choice(VOCAB["openers"])
    if opener:
        parts.append(opener)

    organ_list = VOCAB["organ"].get(row.organ, [])
    if organ_list and rng.random() < 0.85:
        parts.append(rng.choice(organ_list))

    if int(row.is_healthy) == 1:
        parts.append(rng.choice(VOCAB["is_healthy_phrases"]))
        sev_bucket, coverage = 0, 0.0
    else:
        # inclusion probabilities tuned up from the first pass: the weakest
        # heads (margin, distribution, severity) were weak mainly because the
        # corpus too often gave the model zero textual evidence for them, not
        # because MuRIL couldn't learn the phrasing. A farmer describing a
        # lesion in any detail typically gives at least colour + one more axis.
        if row.colour in VOCAB["colour"] and rng.random() < 0.93:
            parts.append(rng.choice(VOCAB["colour"][row.colour]))
        if row.texture in VOCAB["texture"] and rng.random() < 0.70:
            parts.append(rng.choice(VOCAB["texture"][row.texture]))
        if row.margin in VOCAB["margin"] and rng.random() < 0.55:
            parts.append(rng.choice(VOCAB["margin"][row.margin]))
        if row.distribution in VOCAB["distribution"] and rng.random() < 0.55:
            parts.append(rng.choice(VOCAB["distribution"][row.distribution]))

        sev_bucket = int(np.clip(round(row.severity + rng.choice([-1, 0, 0, 0, 1])), 0, 4))
        if rng.random() < 0.85:
            parts.append(rng.choice(VOCAB["severity"][sev_bucket]))

        coverage = float(np.clip(row.coverage * rng.uniform(0.5, 1.6), 0.02, 0.98))
        if rng.random() < 0.5:
            phrase, coverage = nearest_coverage_phrase(coverage)
            parts.append(phrase)

    closer = rng.choice(VOCAB["closers"])
    if closer:
        parts.append(closer)

    text = apply_typos(" ".join(p for p in parts if p), rng)
    return text, sev_bucket, coverage


def main(a):
    rng = random.Random(a.seed)
    rows, qid = [], 0
    for _, cls in PRIORS.iterrows():
        for _ in range(a.n_per_class):
            text, sev, cov = build_sentence(rng, cls)
            rows.append({
                "query_id": qid, "canonical_label": cls.canonical_label,
                "species": cls.species, "text": text, "script_mix": script_mix(text),
                "colour": cls.colour, "margin": cls.margin,
                "distribution": cls.distribution, "texture": cls.texture,
                "organ": cls.organ, "severity": sev, "coverage": cov,
                "is_healthy": int(cls.is_healthy),
            })
            qid += 1

    df = pd.DataFrame(rows).drop_duplicates(subset="text").reset_index(drop=True)

    # per-class 70/15/15 split, same spirit as config.VAL_FRAC / TEST_FRAC
    rng2 = np.random.RandomState(a.seed)
    df["split"] = "train"
    for c in df.canonical_label.unique():
        ci = df.index[df.canonical_label == c].to_numpy().copy()
        rng2.shuffle(ci)
        n = len(ci)
        n_test = max(1, int(n * 0.15))
        n_val = max(1, int(n * 0.15))
        df.loc[ci[:n_test], "split"] = "test"
        df.loc[ci[n_test:n_test + n_val], "split"] = "val"
        df.loc[ci[n_test + n_val:], "split"] = "train"

    out = CORPUS / "query_corpus.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(df):,} unique queries over {df.canonical_label.nunique()} classes -> {out}")
    print(df.split.value_counts().to_string())
    print(df.script_mix.value_counts().to_string())
    print("\nsample rows:")
    print(df.sample(6, random_state=a.seed)[["canonical_label", "text"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_class", type=int, default=340)
    ap.add_argument("--seed", type=int, default=SEED)
    main(ap.parse_args())
