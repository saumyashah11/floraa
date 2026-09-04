# FLORA :: Module 1 — Vision-to-Attribute Encoder

The image side of the project. It does **not** output a disease label directly.
It outputs a structured, pathologist-style symptom description, and the disease
is read off that description. That intermediate representation is what makes
zero-shot recognition possible in Module 2, and what makes counterfactual
explanation nearly free.

---

## 1. Datasets to download

### Automatic (Kaggle API, handled by script 01)

| Purpose | Slug |
|---|---|
| Rose leaves, ~14.9k images | `shuvokumarbasak4004/rose-leaf-disease-dataset` |
| Multi-species flower leaves | `shuvokumarbasak4004/flower-leaf-diseases-dataset-new-and-update` |
| Second rose set (cross-source check) | `warcoder/rose-leaves-disease-detection` |

### Manual, but important

**Mendeley 2vzp6ss7vg** — https://data.mendeley.com/datasets/2vzp6ss7vg/1
Rose, marigold and China rose, 4,479 labelled images. This is the **only**
source with genuine *flower-tissue* disease samples (roughly 487 of them)
rather than leaf-only imagery. Download in a browser, unzip, place under
`/content/flora/raw/mendeley_flowers/`. Without it you cannot claim floral
diagnosis at all.

### Disease coverage: 48 classes, 44 of them disease/disorder states

`class_priors.csv` and `module 2/disease_monographs.yaml` now cover 48
classes (44 disease/disorder states plus 4 healthy states) across ten
species: the original rose/marigold/hibiscus set (22 classes, image data
already local under `mendeley_flowers/`) plus tomato, potato, pepper, apple,
cherry, peach, grape, strawberry, squash and maize (26 classes) drawn from
the PlantVillage-family datasets below. Every one of the 48 classes has a
written symptom monograph, so Module 2's zero-shot text route and Module 3's
free-text symptom analysis both cover the full set, not just the flower
species. **What is not yet done:** the checkpoints under `artifacts/checkpoints/`
were trained before this expansion and only recognise the original 22-class,
three-species set from images. To get image-side detection across all 48
classes you must download the datasets below and rerun the full pipeline
(`01` → `05`, then `module 2/06-08`, then `module 3/09-12`) on the combined
manifest — the code already supports this without modification (`n_disease`,
`SPECIES_LIST` and `ALL_CLASSES` are all derived dynamically from
`class_priors.csv`, nothing is hardcoded to 22). The safest add-on stack is:

1. `vipoooool/new-plant-diseases-dataset` — broad crop coverage, strong for
   leaf-spot, mildew, rust, blight, and healthy states.
2. `spMohanty/plantvillage-dataset` — classic PlantVillage archive for many
   well-labelled leaf diseases.
3. Keep the existing Mendeley set for flower-tissue classes; it is still the
   key source for floral diseases that generic leaf-only datasets do not cover.

Exact Kaggle commands to download them:

```bash
kaggle datasets download -d vipoooool/new-plant-diseases-dataset \
    -p /content/flora/raw/plantvillage_new --unzip

kaggle datasets download -d spMohanty/plantvillage-dataset \
    -p /content/flora/raw/plantvillage --unzip
```

If the goal is to reach 15+ additional disease classes with minimal drift, the
best practical workflow is: download the broader plant datasets, merge them into
`RAW`, review canonical labels in `class_priors.csv`, then retrain
`04_train_module1.py` and `module 3/10_train_module3.py` on the larger label set.

### Pretraining only (script 03)

`vipoooool/new-plant-diseases-dataset` — ~87k images, 38 classes. Used to
adapt the backbone from ImageNet to plant-lesion texture. Never enters the
flower manifest.

```bash
!kaggle datasets download -d vipoooool/new-plant-diseases-dataset \
    -p /content/flora/raw/plantvillage --unzip
```

---

## 2. Run order

```bash
# once per session
from google.colab import drive; drive.mount('/content/drive')
!pip -q install timm==1.0.9 transformers==4.44.2 imagehash==4.3.1 \
    bitsandbytes==0.43.3 accelerate pyyaml scikit-learn tqdm kaggle

# put kaggle.json at /content/drive/MyDrive/FLORA/kaggle.json first
!python 01_download_and_manifest.py --stage discover
#   -> REVIEW manifests/label_map.csv BY HAND. Fill canonical_label.
#      Every value must exist in class_priors.csv or be keep=0.
!python 01_download_and_manifest.py --stage build

!python 02_build_attribute_labels.py                  # class priors, instant
!python 02_build_attribute_labels.py --vlm --limit 6000   # optional refinement

!python 03_pretrain_plantvillage.py --backbone swint --epochs 4   # ~1 session

!python 04_train_module1.py --backbone swint     --mode joint
!python 04_train_module1.py --backbone resnet50  --mode baseline
!python 04_train_module1.py --backbone effnetv2s --mode joint
!python 04_train_module1.py --backbone dinov2s   --mode joint --freeze_backbone

!python 05_evaluate_and_export.py --tag m1_swint_joint
```

---

## 3. The ablation that carries your paper

Run the same backbone in all three modes:

- `--mode baseline` — backbone straight to disease softmax. The conventional system.
- `--mode bottleneck` — disease readable **only** from attribute logits.
- `--mode joint` — both paths.

Your argument is that `bottleneck` loses only a small amount of closed-set
accuracy against `baseline`, and buys interpretability plus the zero-shot
capability Module 2 demonstrates. Report that trade-off honestly; a small
accuracy loss with a large capability gain is a much stronger result than a
fractional accuracy win.

---

## 4. Free-tier operating notes

- **Never unzip onto Drive.** Archives on Drive, images on `/content`. Drive I/O
  will quadruple your epoch time.
- Script 04 checkpoints after **every** epoch and resumes automatically. A
  disconnect costs one epoch, not one session.
- T4 has no bfloat16. Everything here uses fp16 AMP deliberately.
- If you hit OOM: `--batch_size 16`, and set `ACCUM_STEPS = 2` in `config.py`
  to keep the effective batch at 32.
- `dinov2s` is intended as a **frozen** linear probe (`--freeze_backbone`).
  Fine-tuning it end to end will not fit comfortably.
- Expected epoch time on the flower manifest (roughly 20k images, batch 32):
  ResNet-50 about 4 minutes, Swin-Tiny about 7, EfficientNetV2-S about 6,
  frozen DINOv2 about 3.

---

## 5. What Module 2 receives

`results/<tag>_export/` containing `features.npy`, `attr_logits.npy`,
`meta.csv` (predicted descriptors per image) and `classes.json`. The three
`ZSL_UNSEEN` classes in `config.py` never appear in training and are Module 2's
entire evaluation set. Do not train on them, and do not quietly add them back
when accuracy disappoints.
