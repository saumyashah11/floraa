"""Module 3 -- species helpers and cross-module class-set consistency."""
import numpy as np
import torch

from flora_text import (SPECIES_LIST, ALL_CLASSES, SPECIES_OF, species_onehot,
                        species_restricted_mask)
from flora_align import PRIORS


def test_all_classes_matches_class_priors():
    assert ALL_CLASSES == sorted(PRIORS.canonical_label.tolist())


def test_species_onehot_shape_and_index():
    sp = SPECIES_LIST[0]
    oh = species_onehot([sp, sp])
    assert oh.shape == (2, len(SPECIES_LIST))
    assert (oh.sum(dim=-1) == 1).all()
    assert oh[0, SPECIES_LIST.index(sp)] == 1


def test_species_restricted_mask_matches_or_generic():
    classes = ALL_CLASSES[:20]
    sp = SPECIES_LIST[0]
    mask = species_restricted_mask([sp], classes)
    assert mask.shape == (1, len(classes))
    for i, c in enumerate(classes):
        expected = SPECIES_OF[c] in (sp, "mixed")
        assert bool(mask[0, i]) == expected, f"mask wrong for class {c}"


def test_species_restricted_mask_mixed_species_matches_everything_of_its_kind():
    """A species-agnostic 'mixed' class (generic_*) must be selectable
    regardless of which real species is queried -- this is the mechanism the
    webapp relies on for the species-restricted candidate set."""
    mixed_classes = [c for c in ALL_CLASSES if SPECIES_OF[c] == "mixed"]
    assert mixed_classes, "expected at least one 'mixed' (generic_*) class"
    for sp in SPECIES_LIST:
        if sp == "mixed":
            continue
        mask = species_restricted_mask([sp], mixed_classes)
        assert mask.all(), f"mixed classes should all be allowed for species={sp}"
