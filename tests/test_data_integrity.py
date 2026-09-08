"""
Cross-file consistency checks that would otherwise only surface as a
confusing crash deep in a training script (e.g. 02_build_attribute_labels.py
raising SystemExit for a canonical_label with no class_priors.csv row).
These are the checks worth running BEFORE a training pipeline, not after.
"""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent


def load_priors():
    return pd.read_csv(ROOT / "class_priors.csv")


def load_symptom_monographs():
    return yaml.safe_load((ROOT / "module 2" / "disease_monographs.yaml")
                          .read_text(encoding="utf-8"))


def load_treatment_monographs():
    return yaml.safe_load((ROOT / "module 4" / "treatment_monographs.yaml")
                          .read_text(encoding="utf-8"))


def test_class_priors_has_no_duplicate_labels():
    priors = load_priors()
    dupes = priors.canonical_label[priors.canonical_label.duplicated()].tolist()
    assert not dupes, f"duplicate canonical_label rows in class_priors.csv: {dupes}"


def test_class_priors_required_columns():
    priors = load_priors()
    required = {"canonical_label", "species", "colour", "margin", "distribution",
                "texture", "organ", "severity", "coverage", "is_healthy"}
    missing = required - set(priors.columns)
    assert not missing, f"class_priors.csv missing columns: {missing}"


def test_every_class_prior_has_a_symptom_monograph():
    priors = load_priors()
    monographs = load_symptom_monographs()
    missing = set(priors.canonical_label) - set(monographs.keys())
    assert not missing, (
        f"class_priors.csv classes with no disease_monographs.yaml entry: {missing}")


def test_every_symptom_monograph_entry_has_priors_row():
    """The reverse direction: 07_train_alignment.py builds a text bank for
    every monograph key, so an orphaned monograph entry (no matching prior)
    would silently never be trainable/evaluable."""
    priors = load_priors()
    monographs = load_symptom_monographs()
    extra = set(monographs.keys()) - set(priors.canonical_label)
    assert not extra, f"disease_monographs.yaml entries with no class_priors.csv row: {extra}"


def test_symptom_monograph_entries_have_variants():
    monographs = load_symptom_monographs()
    for label, entry in monographs.items():
        assert entry.get("variants"), f"{label} has no 'variants' list"
        assert len(entry["variants"]) >= 1, f"{label} has an empty 'variants' list"


def test_every_class_prior_has_a_treatment_monograph():
    priors = load_priors()
    treatments = load_treatment_monographs()
    missing = set(priors.canonical_label) - set(treatments.keys())
    assert not missing, (
        f"class_priors.csv classes with no treatment_monographs.yaml entry: {missing}")


def test_treatment_monograph_healthy_states_have_no_sources():
    """Healthy states make no factual claims needing citation; disease states
    (this project's sourcing standard) should all carry at least one."""
    priors = load_priors().set_index("canonical_label")
    treatments = load_treatment_monographs()
    unsourced_diseases = [
        label for label, entry in treatments.items()
        if int(priors.loc[label, "is_healthy"]) == 0 and not entry.get("sources")
    ]
    assert not unsourced_diseases, (
        f"disease entries in treatment_monographs.yaml with no 'sources': {unsourced_diseases}")


def test_treatment_sources_have_title_and_url():
    treatments = load_treatment_monographs()
    for label, entry in treatments.items():
        for src in entry.get("sources", []):
            assert "title" in src and "url" in src, f"{label} has a malformed source entry: {src}"
            assert src["url"].startswith("http"), f"{label} source url looks wrong: {src['url']}"


def test_look_alikes_reference_real_classes():
    """A typo'd look_alikes entry would silently just never surface (Module
    4's look_alike_notes() skips unknown keys instead of erroring), so this
    has to be checked here, not at runtime."""
    monographs = load_symptom_monographs()
    all_labels = set(monographs.keys())
    bad = {}
    for label, entry in monographs.items():
        unknown = set(entry.get("look_alikes", [])) - all_labels
        if unknown:
            bad[label] = unknown
    assert not bad, f"look_alikes referencing unknown classes: {bad}"


def test_attribute_schema_matches_class_priors_categorical_values():
    schema = yaml.safe_load((ROOT / "attribute_schema.yaml").read_text(encoding="utf-8"))
    priors = load_priors()
    for head, spec in schema["categorical"].items():
        if head not in priors.columns:
            continue
        allowed = set(spec["values"])
        used = set(priors[head].dropna().unique())
        unknown = used - allowed
        assert not unknown, (
            f"class_priors.csv column '{head}' uses values not in attribute_schema.yaml: {unknown}")
