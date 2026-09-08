"""Module 1 -- FloraAttributeNet shape/mode sanity. pretrained=False so this
never touches the network and stays fast."""
import pytest
import torch

from flora_model import FloraAttributeNet, CAT_HEADS, CAT_VALUES, N_SEVERITY


N_DISEASE = 7
BATCH = 2
IMG_SIZE = 64  # small on purpose -- shape correctness doesn't need real resolution


@pytest.mark.parametrize("mode", ["baseline", "bottleneck", "joint"])
def test_forward_shapes_per_mode(mode):
    model = FloraAttributeNet(backbone="resnet50", n_disease=N_DISEASE, mode=mode,
                              pretrained=False, img_size=IMG_SIZE)
    model.eval()
    x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = model(x, return_feat=True)

    for h in CAT_HEADS:
        assert out[h].shape == (BATCH, len(CAT_VALUES[h])), f"{h} head shape wrong"
    assert out["severity"].shape == (BATCH, N_SEVERITY)
    assert out["coverage"].shape == (BATCH,)
    assert out["is_healthy"].shape == (BATCH,)
    assert out["attr_vec"].shape == (BATCH, model.attr_dim)
    assert out["feat"].shape[0] == BATCH

    if mode in ("bottleneck", "joint"):
        assert out["disease_bottleneck"].shape == (BATCH, N_DISEASE)
    else:
        assert "disease_bottleneck" not in out

    if mode in ("baseline", "joint"):
        assert out["disease_direct"].shape == (BATCH, N_DISEASE)
    else:
        assert "disease_direct" not in out


def test_freeze_backbone_stops_gradients():
    model = FloraAttributeNet(backbone="resnet50", n_disease=N_DISEASE, mode="joint",
                              pretrained=False, freeze_backbone=True, img_size=IMG_SIZE)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    # heads must still train
    assert any(p.requires_grad for p in model.cat_heads.parameters())


def test_attr_dim_matches_concatenated_heads():
    model = FloraAttributeNet(backbone="resnet50", n_disease=N_DISEASE, mode="joint",
                              pretrained=False, img_size=IMG_SIZE)
    expected = sum(len(CAT_VALUES[h]) for h in CAT_HEADS) + N_SEVERITY + 2
    assert model.attr_dim == expected
