# tests/test_unit.py
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import extract_simple_entities, _preprocess_text_for_matching
import zeyrek

# Unit testler için modelleri bir kere yükle
morphology = zeyrek.MorphAnalyzer()

@pytest.mark.parametrize("query, expected_candidate", [
    ("Keten pantolonun fiyatı nedir?", "keten pantolon"),
    ("ipek gömlek hakkında bilgi ver", "ipek gömlek"),
    ("bu deri ceketin malzemesi ne", "deri ceket"),
    ("bana pantolonları göster", "pantolon"),
    ("sadece ceket", "ceket"),
    ("fiyat", ""), 
])
def test_extract_item_candidate(query, expected_candidate):
    lemmatized_query = _preprocess_text_for_matching(query, morphology)
    entities = extract_simple_entities(
        original_query_spell_checked=query,
        processed_query_lemmatized=lemmatized_query,
        current_morphology=morphology
    )
    assert entities.get("item_name_candidate") == expected_candidate

@pytest.mark.parametrize("query, expected_size", [
    ("bunun L bedeni var mı", "L"),
    ("42 numara mevcut mu", "42"),
    ("small beden arıyorum", "SMALL"),
])
def test_extract_size_entity(query, expected_size):
    lemmatized_query = _preprocess_text_for_matching(query, morphology)
    entities = extract_simple_entities(query, lemmatized_query, morphology)
    assert entities.get("size") == expected_size
