"""FLORA :: central configuration. Every other script imports from here."""
import os
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------- paths
# Colab layout kept /content/drive/MyDrive/FLORA (persistent) separate from
# /content/flora (ephemeral, fast local disk). Running locally there is no
# such split, so both point into this project directory unless overridden.
DRIVE = Path(os.environ.get("FLORA_DRIVE", HERE / "artifacts"))
LOCAL = Path(os.environ.get("FLORA_LOCAL", HERE))

RAW = LOCAL                                    # raw/manual datasets live directly here
MANIFESTS = DRIVE / "manifests"
CKPTS = DRIVE / "checkpoints"
CORPUS = DRIVE / "corpus"
RESULTS = DRIVE / "results"
CACHE = DRIVE / "cache"

for _p in (RAW, MANIFESTS, CKPTS, CORPUS, RESULTS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data
KAGGLE_SETS = {
    # slug                                                             -> local folder
    "shuvokumarbasak4004/rose-leaf-disease-dataset":                "rose_leaf",
    "shuvokumarbasak4004/flower-leaf-diseases-dataset-new-and-update": "flower_leaf",
    "warcoder/rose-leaves-disease-detection":                       "rose_warcoder",
    # Broader disease expansion for 15+ additional classes.
    "vipoooool/new-plant-diseases-dataset":                          "plantvillage_new",
    "spMohanty/plantvillage-dataset":                                "plantvillage",
}
# Downloaded separately, used ONLY for backbone adaptation (script 03).
PRETRAIN_SET = ("vipoooool/new-plant-diseases-dataset", "plantvillage")
# Mendeley 2vzp6ss7vg is downloaded by hand -> RAW/"mendeley_flowers"
MANUAL_SETS = ["mendeley_flowers"]

IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp"}

# ---------------------------------------------------------------- split
SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Diseases removed entirely from the image training set. Module 2 must
# recognise these from their text monograph alone. Do not train on them.
# The first three are the paper's real ZSL classes but have zero images in
# the local Mendeley set; rose_black_spot is added as a genuine local
# stand-in so script 08 has real unseen images to score.
ZSL_UNSEEN = [
    "rose_downy_mildew",
    "marigold_botrytis_flower_blight",
    "hibiscus_rust",
    "rose_black_spot",
]

# ---------------------------------------------------------------- train
IMG_SIZE = 224
BATCH_SIZE = 32
ACCUM_STEPS = 1
NUM_WORKERS = 0                                # 0 for local CPU/Windows runs
EPOCHS = 25
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 2
LABEL_SMOOTH = 0.05
PATIENCE = 6

BACKBONES = {
    "resnet50": "resnet50.a1_in1k",
    "effnetv2s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "swint": "swin_tiny_patch4_window7_224.ms_in1k",
    "dinov2s": "vit_small_patch14_dinov2.lvd142m",
}

LOSS_WEIGHTS = {
    "colour": 1.0, "margin": 1.0, "distribution": 1.0,
    "texture": 1.0, "organ": 0.7,
    "severity": 0.7, "coverage": 0.3, "is_healthy": 0.5,
    "disease": 1.0,
}
