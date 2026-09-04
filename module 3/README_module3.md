# FLORA :: Module 3 — Farmer Query Understanding

Modules 1 and 2 assume the system receives a photograph. Real farmers send a
photograph *and* a sentence like "patti pe daag aa gaye, pandhre pandhre
dikhtay" — Hindi, Marathi, English and phonetic spelling mixed in one breath,
often with no punctuation and no clean grammar. Module 3 reads that sentence
and turns it into the exact same symptom vocabulary Module 1 predicts from
pixels (`attribute_schema.yaml`: colour, margin, distribution, texture,
organ, severity, coverage, is_healthy), then fuses the two before a diagnosis
is read off.

Every number quoted below came out of an actual run of these five scripts on
this machine, against `m1_resnet50_joint`'s real predictions on real Mendeley
photographs. Nothing is simulated except the query text itself, and that is
built the same way Module 1's colour/margin/distribution/texture labels are
built: programmatically from `class_priors.csv`, not by hand.

There is also a local web app (`webapp/`) that runs this pipeline live —
see [§5](#5-the-local-web-app-webapp) — kept deliberately separate from any
hosted demo: it is a real Flask process on your machine running the actual
trained checkpoints, not a static page.

---

## Run order

```bash
pip install -q transformers torch pyyaml pandas scikit-learn matplotlib flask

python "module 3/09_build_query_corpus.py" --n_per_class 340
python "module 3/10_train_module3.py"      --encoder muril --epochs 90 --hidden 320
python "module 3/11_fuse_and_evaluate.py"  --encoder muril --noise_max 3.0
python "module 3/12_make_report.py"        --encoder muril

python "module 3/webapp/app.py"    # -> http://127.0.0.1:5050
```

Script 10 downloads `google/muril-base-cased` once (~950MB) and caches its
frozen sentence embeddings; every later run of 10 or 11 reads the cache and
finishes in seconds. On this machine, embedding the full 6,982-query corpus
took ~5 minutes on CPU and head training took under 2 minutes for 90 epochs.
`--encoder` also accepts `indicbert`, `marathi_bert`, `hi_mr_dev` — see
`TEXT_ENCODERS` in `flora_text.py`.

---

## The pipeline

```
farmer's sentence                real photograph (Module 1)
        │                                   │
   MuRIL (frozen)                  ResNet-50 backbone (frozen, trained)
        │                                   │
FloraTextAttributeNet              FloraAttributeNet attribute heads
        │  attr_vec [38]                    │  attr_vec [38]
        └──────────────┬────────────────────┘
                        ▼
                  FusionGate
        (16 confidence numbers in, 1 blend weight out)
                        ▼
              fused attribute vector ──────────────┐
                        │                           │
                        ▼                           ▼
      nearest class prototype              DiseaseHead (species-conditioned,
      (class_priors.csv, reused                learned classifier)
      from Module 2's flora_align.py)             │
                        │                          │
                        ▼                          ▼
              disease name                   disease name
        (zero-shot capable,             (far higher accuracy,
         needs no training                but blind to a class
         examples of the class)          it never saw a label for)
```

`flora_text.py` mirrors Module 1's head layout exactly — same `CAT_HEADS`,
same `SPANS`, same `normalise_attr` — so the two attribute vectors are
directly comparable, axis for axis, before fusion ever happens. Two
read-off routes exist deliberately, not as leftover scaffolding: they trade
off in opposite directions and the evaluation reports both rather than
picking a winner and hiding the loser.

---

## 1. The corpus (`09_build_query_corpus.py`)

`query_vocabulary.yaml` is the farmer-speech mirror of `attribute_schema.yaml`
— Hindi, Marathi and Hinglish phrasings for every colour/margin/distribution/
texture/organ/severity value, plus opener/closer chatter, coverage phrases,
and a typo pool. The builder samples a random *subset* of axes per sentence
(no farmer reports margin shape and lesion distribution in the same breath),
perturbs severity and coverage per row, and injects spelling noise — **6,982**
unique queries survive deduplication over 22 classes, split 70/15/15 like the
image manifest:

| script mix | share |
|---|---|
| mixed Devanagari + Latin in one sentence | 88% |
| pure Hinglish (Latin transliteration) | 12% |
| pure Devanagari | <1% |

That mixed-script majority is not an artefact — it is what real Hindi/Marathi
chat input looks like, and MuRIL has to learn through it. (A first pass used
4,586 queries with lower per-axis mention rates; the weakest heads — margin,
distribution, severity — were weak mainly because the corpus too often gave
the model *zero textual evidence* for them, not because MuRIL couldn't learn
the phrasing. Raising the inclusion probabilities and adding more phrase
variants, then regenerating a 52% larger corpus, is what closed most of that
gap — see the table below.)

## 2. Text attribute heads (`10_train_module3.py`)

Frozen MuRIL embeddings, lightweight heads on top — Module 1's frozen-DINOv2
linear-probe pattern applied to text, for the same reason: a few thousand
short sentences will let a 240M-parameter encoder memorise the corpus outright
if it is fine-tuned end to end, and the held-out numbers stop meaning
anything. Held-out test accuracy, before and after the corpus/capacity pass:

| head | v1 (4.6k queries, hidden=256) | v2 (7.0k queries, hidden=320) |
|---|---|---|
| colour | 0.975 | **0.985** |
| margin | 0.948 | **0.963** |
| distribution | 0.926 | **0.941** |
| texture | 0.949 | **0.962** |
| organ | 0.981 | **0.990** |
| severity | 0.878 | **0.943** |
| is_healthy | 1.000 | 0.998 |
| coverage (MAE, lower better) | 0.077 | 0.078 |

Severity moved the most (+6.5 points) because it was the head most starved of
textual evidence in v1 — see `m3_attribute_accuracy.png`.

## 3. Fusion and the two read-off routes (`11_fuse_and_evaluate.py`)

Image evidence is `m1_resnet50_joint_full_export`'s real `attr_logits.npy` on
Module 1's own held-out splits — `val` (478 images) trains `FusionGate` and
`DiseaseHead`, `test` (449 images, 7 classes Module 1 was trained on) and
`zsl_test` (554 images, all `rose_black_spot` — the one class **removed from
Module 1's image training set entirely**) report the numbers. Text is paired
in by matching `canonical_label` against Module 3's own disjoint corpus
splits — nothing trains and evaluates on the same pairing.

`FusionGate` is trained with random noise injected on the image side (up to
σ=3 on raw logits) so it learns to fall back on text when a photo is bad.
**`DiseaseHead`** is new: a small classifier trained *on top of* the frozen
gate's fused vector plus a one-hot plant species (`flora_text.DiseaseHead`),
replacing the earlier all-nearest-prototype read-off with something that can
actually be trained. The reason to add it: nearest-prototype matching is an
*unlearned* heuristic — cosine similarity to a one-hot class prior cannot
exploit anything beyond raw attribute agreement, so two classes sharing a
`class_priors.csv` row (a species-specific disease and its `generic_*`
counterpart) were permanently indistinguishable to it, which is why the first
pass's seen-class accuracy topped out around 20–44%.

The trade-off is real and the evaluation reports it honestly rather than
picking whichever route looks better:

| route | needs training examples? | seen-class accuracy | zero-shot capable? |
|---|---|---|---|
| `fusion_gate` (nearest prototype) | no — just a written attribute profile | moderate | **yes** |
| `fusion_learned` (`DiseaseHead`) | yes | **very high** | no |

Two candidate-set protocols are still reported side by side (open-set over
all 22 classes vs. species-restricted, the realistic "app already knows the
crop" setting):

| protocol | subset | image only | text only | fusion (confidence) | fusion (prototype) | **fusion (learned)** |
|---|---|---|---|---|---|---|
| open_set_22 | seen_test (n=449) | 0.196 | 0.129 | 0.205 | 0.205 | **0.987** |
| open_set_22 | zsl_unseen_test (n=554) | 0.000 | 0.875 | 0.323 | **0.773** | 0.000 |
| species_restricted | seen_test (n=449) | 0.401 | 0.325 | 0.425 | 0.419 | **0.984** |
| species_restricted | zsl_unseen_test (n=554) | 0.000 | 0.875 | 0.323 | **0.773** | 0.000 |

Read both extreme rows literally. On `seen_test`, `fusion_learned` reaches
**98.4–98.7%** — a real classifier exploiting correlations the prior table
throws away, not a heuristic anymore. On `zsl_unseen_test`
(`rose_black_spot`, never in Module 1's image training and never in the
val split `DiseaseHead` trained on), `fusion_learned` is **exactly 0%** — it
was never given a positive label for that class in any modality, so it
structurally cannot predict it, no matter how good the input. `fusion_gate`
(nearest prototype) is the opposite: needing zero examples, it actually
*improved* here too (65.5% → 77.3%) purely from the better-trained text
encoder feeding it cleaner attribute vectors. **Neither route dominates —
use `fusion_learned` for anything with training data in either modality, and
keep `fusion_gate` as the fallback for a disease nobody has photographed
yet.** `m3_confusion_fusion.png` shows this as three panels side by side:
image-only is diffuse everywhere, `fusion_learned` is a near-perfect diagonal
on the 7 trained classes and *empty* on `rose_black_spot`, `fusion_gate` is
moderate everywhere but is the only one with mass on that row at all.

### Robustness to bad photos

Both fusion routes are evaluated as synthetic noise is added to the image
logits — simulating blur, bad light, a half-visible leaf (species-restricted
protocol; `DiseaseHead` uses the same frozen, noise-trained gate, so its
input already adapts even though the head itself wasn't retrained per noise
level):

| noise σ | image only | fusion (confidence) | fusion (prototype) | fusion (learned) |
|---|---|---|---|---|
| 0.0 | 0.401 | 0.425 | 0.419 | **0.987** |
| 1.0 | 0.365 | 0.402 | 0.405 | **0.987** |
| 3.0 | 0.276 | 0.385 | 0.376 | **0.988** |
| 6.0 | 0.138 | 0.365 | 0.355 | **0.990** |
| 9.0 | 0.099 | 0.356 | 0.357 | **0.987** |

`fusion_learned` barely moves across the entire sweep — once the gate has
downweighted a noisy photo, the classifier is effectively reading a
mostly-text signal it was trained to be very good at. Image-only collapses by
a factor of four. See `m3_robustness_curve.png`.

### A concrete example (real, from `m3_qualitative_examples.csv`)

Subset `zsl_unseen_test`, true label `rose_black_spot`:

> *"photo bhej raha hoon, dekhiye patti par black spot jaisa kuch sapaट hai,
> koi ubhaar nahi पत्तीभर विखुरलेले डाग kuch hisse mein काय करावं सांगा"*

Image-only prediction: `marigold_pest_damage` (wrong — Module 1 has never
seen this disease's photos). Text-only / `fusion_gate`: `rose_black_spot`
(correct). `fusion_learned`, tested live through the web app on the same
sentence, confidently says `rose_pest_damage` (**wrong**, 99.5% confidence)
— it has simply never been shown a `rose_black_spot` label. This is the
trade-off from the table above, reproduced live, not just in a spreadsheet.

---

## 4. Files

- `query_vocabulary.yaml` — the farmer-speech vocabulary, mirroring
  `attribute_schema.yaml`.
- `flora_text.py` — `FloraTextAttributeNet`, `FusionGate`, `DiseaseHead`,
  `confidence_weighted_fusion`, `predict_disease`, `species_onehot`,
  `species_restricted_mask`. Imports `SPANS`, `normalise_attr` and
  `build_attr_prototypes` directly from Module 2's `flora_align.py` rather
  than redefining them.
- `09_build_query_corpus.py` → `artifacts/corpus/query_corpus.csv`.
- `10_train_module3.py` → `artifacts/checkpoints/m3_text_<encoder>_best.pt`,
  `artifacts/results/m3_text_<encoder>_{history,test_report}.json`.
- `11_fuse_and_evaluate.py` → `artifacts/checkpoints/m3_fusion_gate_best.pt`,
  `m3_disease_head_best.pt`, `m3_route_comparison.csv`,
  `m3_robustness_sweep.csv`, `m3_confusion.npz` (three matrices:
  `image_only`, `fusion_gate`, `fusion_learned`),
  `m3_qualitative_{examples,full}.csv`, `m3_fusion_report.json`.
- `12_make_report.py` → seven PNGs in `artifacts/results/`:
  `m3_corpus_composition`, `m3_training_curves`, `m3_attribute_accuracy`,
  `m3_route_comparison`, `m3_robustness_curve`, `m3_confusion_fusion`,
  `m3_gate_weight_distribution`.
- `webapp/` — the local app, see below.

## 5. The local web app (`webapp/`)

```bash
pip install flask
python "module 3/webapp/app.py"
# -> http://127.0.0.1:5050
```

A real Flask server, not a static export. It loads the actual checkpoints —
MuRIL, `FloraTextAttributeNet`, `FusionGate`, `DiseaseHead` — once at
startup and runs genuine live inference on whatever is typed into it:

- **Symptom decoder** — type a message, optionally attach a real photograph
  from the held-out test set (served straight off disk via `/media/<id>`)
  and pick a plant species if known. `/api/decode` runs the full pipeline
  live and returns both fusion routes side by side, so a disagreement
  between them is visible rather than papered over.
- **Real results** — the same charts described above, rendered as inline SVG
  from `/api/report`, which reads the actual files scripts 09–12 wrote to
  `artifacts/results/`.
- **Case by case** — the qualitative gallery, image/text/both-fusion-routes
  predictions side by side per example.

This app requires the pipeline above to have been run at least once (it
loads checkpoints and result files that don't exist otherwise). It is
intentionally separate from any hosted demo of this project: the interactive
"try it" in a sandboxed static page has no way to run a 240M-parameter model,
so it would have to fake the encoder with keyword matching; this app has a
real Python process behind it and never approximates.

## What would make this stronger

The obvious next ablations, in the style Module 2's README asks for: sweep
`--encoder` across `indicbert`/`marathi_bert`; retrain `FusionGate` with
`--noise_max 0` to confirm the noise augmentation is what buys the
robustness curve rather than the architecture alone; and train `DiseaseHead`
with a few synthetic zero-image rows per untrained class (text evidence
only, paired with a null image vector) to see whether it can be taught to
at least *consider* a class it has no photos of, instead of having zero
probability mass on it by construction. The corpus itself is still
templated: real transcribed farmer queries from a pilot, once any exist,
would matter more than any further code change here. And Module 1's own
image-classification ceiling (visible in the `image_only` row throughout)
is set by its Colab training run on datasets not present in this local
checkout (`rose_leaf_disease-dataset`, `flower-leaf-diseases-dataset`) —
improving it is a Module 1 retraining job, not a Module 3 one.
