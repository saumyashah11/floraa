# FLORA :: Module 2 — Language-Space Diagnostic Reasoner

Module 1 turns an image into a symptom description. Module 2 reasons over that
description against a corpus of disease monographs, and thereby names diseases
whose images the system has **never seen**. This is the research contribution;
everything else is scaffolding around it.

**Corpus size:** `disease_monographs.yaml` covers all 48 classes now listed in
`class_priors.csv` (44 disease/disorder states, 4 healthy states, ten
species). The evaluation numbers quoted later in this file are from the
original nineteen-seen / three-ZSL flower-only run and predate the expansion
to the crop diseases; rerun `06`–`08` on a manifest built from the full
`class_priors.csv` to get numbers for the larger class set.

---

## Run order

```bash
!pip -q install sentence-transformers==3.0.1 transformers==4.44.2 pyyaml

!python 06_extract_embeddings.py --tag m1_swint_joint
!python 07_train_alignment.py    --tag m1_swint_joint --encoder mpnet
!python 08_zsl_eval.py           --tag m2_m1_swint_joint_mpnet
```

Scripts 07 and 08 operate on precomputed vectors, not images. A full training
run is two to four minutes on a T4 and works on CPU. You can afford to sweep
encoders and hyperparameters freely, which is unusual on free tier and worth
exploiting.

---

## The three routes

| Route | Image side | Text side | Can name an unseen disease? |
|---|---|---|---|
| `proto` | normalised attribute vector | one-hot prior from `class_priors.csv` | Yes, but only what the prior table already encodes |
| `text` | projected attribute vector | projected monograph embedding | **Yes, from prose alone** |
| `fusion` | z-scored combination of both | | Yes, usually strongest |

The `text` route is the claim. `proto` is the control that proves the text
tower contributes something the structured priors do not.

---

## Evaluation protocol

Script 08 uses the **generalised** zero-shot protocol, not the soft version.
The candidate set at test time contains all nineteen classes, and the test pool
mixes held-out images of seen classes with every image of the unseen classes.
It reports per-class-mean seen accuracy, per-class-mean unseen accuracy, their
harmonic mean H, and AUSUC across a calibration sweep.

Report **H**. Unseen-only accuracy with a restricted candidate set is the number
most zero-shot papers quietly present, and an examiner who knows the field will
ask which one you computed. Being able to answer "generalised, with calibrated
stacking, and here is the full sweep curve" is worth more than a higher number.

---

## Ablations that belong in the paper

```bash
# encoder comparison
for e in mpnet bge minilm scibert; do
  python 07_train_alignment.py --tag m1_swint_joint --encoder $e
  python 08_zsl_eval.py --tag m2_m1_swint_joint_$e
done

# the bottleneck-leak control
python 07_train_alignment.py --tag m1_swint_joint --encoder mpnet --use_features
python 08_zsl_eval.py --tag m2_m1_swint_joint_mpnet_feat

# module 1 mode sweep, propagated all the way through
python 06_extract_embeddings.py --tag m1_swint_bottleneck
python 07_train_alignment.py    --tag m1_swint_bottleneck --encoder mpnet
```

The `--use_features` run is the most important one. It lets the image tower see
raw backbone features and bypass the attribute bottleneck. Seen-class accuracy
will rise and unseen-class accuracy should **fall**, because backbone features
carry no information about a class the backbone never saw. If unseen accuracy
does not fall, something is leaking and you must find it before writing
anything up.

---

## Two failure modes, and what they mean

**`proto` beats `text` on unseen classes.** Your monographs contain no symptom
information beyond what `class_priors.csv` already encodes, so the text tower is
a slower reimplementation of a lookup table. Fix by rewriting the monographs
with discriminative detail the prior table cannot express: lesion progression
over time, which organ is affected first, what the disease is confused with,
weather conditions that drive it.

**Unseen accuracy near zero with high seen accuracy.** This is normal before
calibration and is exactly what calibrated stacking exists to correct. Check
the `_sweep.csv` file: if H peaks at a sensible gamma, the model is working and
was simply seen-biased. If H stays near zero across the whole sweep, the unseen
monographs are not discriminative and the look-alike classes are absorbing them.
The per-unseen-class breakdown table tells you which class is stealing the
predictions.

---

## Files

- `disease_monographs.yaml` — the knowledge base. 48 classes, at least two
  prose variants each, plus an auto-generated variant derived from the priors.
- `flora_align.py` — attribute layout, prototypes, text tower, projection
  towers, alignment objective.
- `06_extract_embeddings.py` — Module 1 inference over every split.
- `07_train_alignment.py` — the alignment training loop.
- `08_zsl_eval.py` — GZSL evaluation, calibration sweep, retrieval, ablation.

Module 3 consumes `flora_align.py` directly: the farmer's free-text complaint is
encoded by the same text tower and fused with the image attribute vector in the
same shared space, which is why the projection dimension is a shared constant.
