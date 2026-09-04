"""FLORA :: the attribute-bottleneck network, its dataset, and its transforms."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from config import BACKBONES, IMG_SIZE, LOSS_WEIGHTS

ImageFile.LOAD_TRUNCATED_IMAGES = True
HERE = Path(__file__).parent
SCHEMA = yaml.safe_load((HERE / "attribute_schema.yaml").read_text())
CAT_HEADS = list(SCHEMA["categorical"].keys())
CAT_VALUES = {h: SCHEMA["categorical"][h]["values"] for h in CAT_HEADS}
N_SEVERITY = SCHEMA["ordinal"]["severity"]["n_bins"]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transforms(train: bool, size: int = IMG_SIZE):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.2),
            # Field photos arrive at arbitrary phone tilt and wildly variable
            # exposure -- neither is covered by crop/flip/colour-jitter alone.
            T.RandomApply([T.RandomRotation(25)], p=0.5),
            T.RandomApply([T.ColorJitter(0.25, 0.25, 0.20, 0.03)], p=0.7),
            T.RandomAutocontrast(p=0.2),
            T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.15),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.20, scale=(0.02, 0.12)),
        ])
    return T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class FloraDataset(Dataset):
    """Reads manifest_attr.csv. Returns image plus every attribute target."""

    def __init__(self, df: pd.DataFrame, disease_classes, train=True, size=IMG_SIZE):
        self.df = df.reset_index(drop=True)
        self.tf = build_transforms(train, size)
        self.disease_to_idx = {c: i for i, c in enumerate(disease_classes)}
        self.cat_to_idx = {h: {v: i for i, v in enumerate(CAT_VALUES[h])} for h in CAT_HEADS}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            img = Image.open(r.image_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        x = self.tf(img)
        y = {h: torch.tensor(self.cat_to_idx[h].get(str(r[h]), 0), dtype=torch.long)
             for h in CAT_HEADS}
        y["severity"] = torch.tensor(int(r.severity), dtype=torch.long)
        y["coverage"] = torch.tensor(float(r.coverage), dtype=torch.float32)
        y["is_healthy"] = torch.tensor(float(r.is_healthy), dtype=torch.float32)
        y["disease"] = torch.tensor(
            self.disease_to_idx.get(r.canonical_label, -100), dtype=torch.long)
        return x, y


class FloraAttributeNet(nn.Module):
    """
    mode='baseline'   : backbone -> disease softmax. The thing everyone builds.
    mode='bottleneck' : backbone -> attribute heads -> disease read ONLY from
                        the concatenated attribute logits. Interpretable, and
                        the reason zero-shot transfer works downstream.
    mode='joint'      : both paths, losses summed. Usually the best accuracy.
    """

    def __init__(self, backbone="swint", n_disease=14, mode="joint",
                 pretrained=True, freeze_backbone=False, img_size=IMG_SIZE,
                 drop_rate=0.2):
        super().__init__()
        self.mode = mode
        name = BACKBONES.get(backbone, backbone)
        kw = dict(pretrained=pretrained, num_classes=0)
        if "dinov2" in name:
            kw.update(img_size=img_size, dynamic_img_size=True)
        self.backbone = timm.create_model(name, **kw)
        d = self.backbone.num_features
        self.drop = nn.Dropout(drop_rate)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.cat_heads = nn.ModuleDict(
            {h: nn.Linear(d, len(CAT_VALUES[h])) for h in CAT_HEADS})
        self.severity_head = nn.Linear(d, N_SEVERITY)
        self.coverage_head = nn.Linear(d, 1)
        self.healthy_head = nn.Linear(d, 1)

        self.attr_dim = sum(len(CAT_VALUES[h]) for h in CAT_HEADS) + N_SEVERITY + 2
        self.bottleneck_head = nn.Sequential(
            nn.Linear(self.attr_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, n_disease))
        self.direct_head = nn.Linear(d, n_disease)

    def forward(self, x, return_feat=False):
        f = self.drop(self.backbone(x))
        out = {h: self.cat_heads[h](f) for h in CAT_HEADS}
        out["severity"] = self.severity_head(f)
        out["coverage"] = self.coverage_head(f).squeeze(-1)
        out["is_healthy"] = self.healthy_head(f).squeeze(-1)

        attr_vec = torch.cat(
            [out[h] for h in CAT_HEADS]
            + [out["severity"], out["coverage"].unsqueeze(-1), out["is_healthy"].unsqueeze(-1)],
            dim=-1)
        out["attr_vec"] = attr_vec
        if self.mode in ("bottleneck", "joint"):
            out["disease_bottleneck"] = self.bottleneck_head(attr_vec)
        if self.mode in ("baseline", "joint"):
            out["disease_direct"] = self.direct_head(f)
        if return_feat:
            out["feat"] = f
        return out


def flora_loss(out, y, label_smooth=0.05, class_weight=None):
    """Weighted multi-task objective. Returns (total, per-term dict)."""
    parts = {}
    for h in CAT_HEADS:
        parts[h] = F.cross_entropy(out[h], y[h], label_smoothing=label_smooth)
    parts["severity"] = F.cross_entropy(out["severity"], y["severity"],
                                        label_smoothing=label_smooth)
    parts["coverage"] = F.smooth_l1_loss(torch.sigmoid(out["coverage"]), y["coverage"])
    parts["is_healthy"] = F.binary_cross_entropy_with_logits(
        out["is_healthy"], y["is_healthy"])

    dis = 0.0
    for k in ("disease_bottleneck", "disease_direct"):
        if k in out:
            dis = dis + F.cross_entropy(out[k], y["disease"], ignore_index=-100,
                                        weight=class_weight,
                                        label_smoothing=label_smooth)
    if isinstance(dis, torch.Tensor):
        parts["disease"] = dis

    total = sum(LOSS_WEIGHTS.get(k, 1.0) * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}


@torch.no_grad()
def severity_expectation(logits):
    """Ordinal-aware point estimate, for reporting MAE rather than accuracy."""
    p = torch.softmax(logits, dim=-1)
    bins = torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)
    return (p * bins).sum(-1)
