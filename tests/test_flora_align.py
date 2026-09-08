"""Module 2 -- attribute layout and prototype-building sanity."""
import torch

from flora_align import SPANS, ATTR_DIM, normalise_attr, build_attr_prototypes, PRIORS
from flora_model import CAT_HEADS


def test_spans_cover_attr_dim_exactly_once():
    covered = torch.zeros(ATTR_DIM, dtype=torch.bool)
    for name, (s, e) in SPANS.items():
        assert not covered[s:e].any(), f"span '{name}' overlaps an earlier span"
        covered[s:e] = True
    assert covered.all(), "SPANS leaves gaps in the attribute vector"


def test_normalise_attr_categorical_heads_sum_to_one():
    x = torch.randn(3, ATTR_DIM)
    out = normalise_attr(x)
    for h in list(CAT_HEADS) + ["severity"]:
        s, e = SPANS[h]
        sums = out[:, s:e].sum(-1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5), f"{h} softmax doesn't sum to 1"


def test_normalise_attr_scalars_are_bounded():
    x = torch.randn(3, ATTR_DIM) * 10
    out = normalise_attr(x)
    for h in ("coverage", "is_healthy"):
        s, _ = SPANS[h]
        vals = out[:, s]
        assert (vals >= 0).all() and (vals <= 1).all(), f"{h} sigmoid output out of [0,1]"


def test_build_attr_prototypes_shape_and_onehot_structure():
    classes = sorted(PRIORS.canonical_label.unique().tolist())[:5]
    proto = build_attr_prototypes(classes)
    assert proto.shape == (len(classes), ATTR_DIM)
    # every categorical head must be exactly one-hot per class
    for h in CAT_HEADS:
        s, e = SPANS[h]
        nnz = (proto[:, s:e] != 0).sum(dim=-1)
        assert (nnz == 1).all(), f"prototype head '{h}' is not one-hot for all classes"


def test_build_attr_prototypes_unknown_class_raises():
    import pytest
    with pytest.raises(KeyError):
        build_attr_prototypes(["not_a_real_class_xyz"])
