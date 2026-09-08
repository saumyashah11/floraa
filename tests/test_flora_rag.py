"""
Module 4 -- corpus loading, retrieval filtering, look-alike notes, and
prompt construction. None of these need the embedding or generation models
loaded, so they stay fast and don't depend on network/HF cache state.
"""
import torch
import pytest

from flora_rag import (load_corpus, retrieve, build_prompt, look_alike_notes,
                       wants_hindi_reply, SECTIONS, CONFIDENCE_THRESHOLD)
from flora_align import PRIORS


def test_load_corpus_covers_every_class_priors_row():
    passages = load_corpus()
    labels = {p["label"] for p in passages}
    assert labels == set(PRIORS.canonical_label), (
        "treatment corpus doesn't cover exactly the classes in class_priors.csv")


def test_load_corpus_sections_are_known():
    passages = load_corpus()
    assert passages, "corpus is empty"
    for p in passages:
        assert p["section"] in SECTIONS
        assert p["text"], f"empty passage text for {p['label']}/{p['section']}"


def test_retrieve_filters_to_requested_disease():
    passages = load_corpus()
    emb = torch.randn(len(passages), 8)  # content doesn't matter, only filtering does
    disease = passages[0]["label"]
    top = retrieve(passages, emb, disease=disease, k=100)
    assert top, "retrieve returned nothing for a real disease"
    assert all(p["label"] == disease for p in top)


def test_retrieve_unknown_disease_raises():
    passages = load_corpus()
    emb = torch.randn(len(passages), 8)
    with pytest.raises(KeyError):
        retrieve(passages, emb, disease="not_a_real_class_xyz")


def test_retrieve_respects_k():
    passages = load_corpus()
    emb = torch.randn(len(passages), 8)
    disease = passages[0]["label"]
    top = retrieve(passages, emb, disease=disease, k=1)
    assert len(top) == 1


def test_look_alike_notes_reference_a_different_class():
    passages = load_corpus()
    labels_with_alikes = None
    for p in passages:
        notes = look_alike_notes(p["label"])
        if notes:
            labels_with_alikes = (p["label"], notes)
            break
    assert labels_with_alikes, "no class in the corpus has look_alikes -- unexpected"
    label, notes = labels_with_alikes
    for n in notes:
        assert n["look_alike"] != label, "a class listed as its own look-alike"
        assert n["look_alike_symptoms"], "empty look-alike symptom text"


def test_look_alike_notes_empty_for_unknown_class():
    assert look_alike_notes("not_a_real_class_xyz") == []


def test_build_prompt_includes_look_alike_instruction_only_when_given():
    passages = [{"section": "cultural", "text": "remove infected leaves"}]
    plain = build_prompt("Test disease", passages)
    assert "look-alike" not in plain.lower()

    notes = [{"look_alike_common_name": "Other disease",
              "look_alike_symptoms": "different spots",
              "this_disease_symptoms": "some spots"}]
    with_alikes = build_prompt("Test disease", passages, look_alikes=notes)
    assert "Other disease" in with_alikes
    assert "look-alike" in with_alikes.lower()


def test_confidence_threshold_is_a_probability():
    assert 0.0 < CONFIDENCE_THRESHOLD < 1.0


def test_wants_hindi_reply_detects_devanagari_and_hinglish():
    assert wants_hindi_reply("पत्ते पर काले धव्बे हैं")  # patte par kaale dhabbe hain
    assert wants_hindi_reply("patti par kaale dhabbe hain, ilaj bataiye please")
    assert not wants_hindi_reply("is it too late to save my plants?")
    assert not wants_hindi_reply(None)
    assert not wants_hindi_reply("")


def test_build_prompt_is_always_english():
    """Regression guard: an earlier approach asked the generator to write
    natively in Hindi and got incoherent Devanagari text back. The fix is a
    separate translation pass (translate_text), not a prompt instruction --
    build_prompt must never mention a reply language at all, regardless of
    the farmer's query language, since generation always stays English."""
    passages = [{"section": "cultural", "text": "remove infected leaves"}]
    prompt = build_prompt("Test disease", passages, query="patti par daag hai, ilaj bataiye")
    assert "hindi" not in prompt.lower()

    prompt_devanagari_query = build_prompt("Test disease", passages,
                                           query="पत्ते पर दाग है")
    assert "hindi" not in prompt_devanagari_query.lower()


def test_translate_text_english_is_a_noop():
    from flora_rag import translate_text
    assert translate_text("hello", language="en") == "hello"
    assert translate_text("", language="hi") == ""
    assert translate_text(None, language="hi") is None


def test_translate_text_unknown_language_raises():
    from flora_rag import translate_text
    import pytest
    with pytest.raises(ValueError):
        translate_text("hello", language="xx")
