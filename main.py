import sys
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from pydantic import BaseModel, Field
import re
import fasttext
import zeyrek
from rapidfuzz import process, fuzz
from pathlib import Path
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
from contextlib import asynccontextmanager
import os

from symspellpy import SymSpell, Verbosity

# Proje kök dizinini doğru bir şekilde belirleyelim
BASE_DIR = Path(__file__).resolve().parent

# Loglama ayarları
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic Modelleri
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class NLUSingleAnalysis(BaseModel):
    slm_intent: str
    slm_entities: List[Dict[str, Any]]
    confidence_score: float
    message: str

class NLUResponse(BaseModel):
    original_query: str
    processed_query_for_nlu: Optional[str] = None
    session_id: str
    nlu_method: str
    analysis: Optional[NLUSingleAnalysis] = None
    detected_intent: Optional[str] = None
    previous_query_in_session: Optional[str] = None
    resolved_product: Optional[str] = None
    resolved_size: Optional[str] = None
    actionable_message: Optional[str] = None
    bot_response: Optional[str] = None
    ask_for_clarification: bool = False
    clarification_options: Optional[List[str]] = None

# Simüle Edilmiş İşletme Verileri
simulated_business_data = {
    "business_info": {
        "name": "Mantarinyo Butik", "phone": "0555 123 4567",
        "address": "Moda Caddesi No:1 Kadıköy, İstanbul", "website": "http://www.mantarinyo-butik.com",
        "email": "info@mantarinyo-butik.com", "maps_link": "https://maps.app.goo.gl/örnekLink",
        "whatsapp_number": "0555 123 4567",
        "shipping_info": "Siparişleriniz Yurtiçi Kargo ile 2-3 iş gününde teslim edilir. Kargo ücreti 50 TL'dir, 500 TL ve üzeri alışverişlerde kargo ücretsizdir.",
        "return_policy": "Ürünleri teslim aldıktan sonra 14 gün içinde, kullanılmamış ve etiketi üzerinde olmak kaydıyla faturanızla birlikte iade edebilirsiniz. İade kargo ücreti müşteriye aittir.",
        "opening_hours": "Haftaiçi: 09:00-19:00, Cumartesi: 10:00-18:00, Pazar: Kapalı",
        "payment_options": "Kredi Kartı (Visa, Mastercard, Amex), Banka Kartı, Havale/EFT kabul edilmektedir."
    },
    "products": {
        "Keten Pantolon": {"link": "http://www.mantarinyo-butik.com/keten-pantolon", "price": "850 TL", "description": "Yüksek kaliteli, %100 doğal ketenden üretilmiş, yaz ayları için ideal, rahat kesim pantolon.", "available_sizes_info": "S, M, L, XL ve 38-46 arası bedenler genellikle bulunur. Detaylar ürün sayfasında.", "material_composition": "%100 Keten", "category": "pantolon"},
        "İpek Gömlek": {"link": "http://www.mantarinyo-butik.com/ipek-gomlek", "price": "1250 TL", "description": "%100 saf ipekten üretilmiş, özel günler ve günlük kullanım için şık gömlek.", "available_sizes_info": "S, M, L bedenleri mevcuttur. Beden tablosu için linke tıklayın.", "material_composition": "%100 İpek", "category": "gömlek"},
        "Deri Ceket": {"link": "http://www.mantarinyo-butik.com/deri-ceket", "price": "3200 TL", "description": "Hakiki kuzu derisinden, klasik motorcu kesim, uzun ömürlü ceket.", "available_sizes_info": "M, L, XL, XXL. Detaylar ürün sayfasında.", "material_composition": "Hakiki Kuzu Derisi, İç Astar: Pamuk", "category": "ceket"},
        "Kot Pantolon": {"link": "http://www.mantarinyo-butik.com/kot-pantolon", "price": "950 TL", "description": "Rahat kesim, dayanıklı kot pantolon.", "available_sizes_info": "28-38 beden arası. Detaylar ürün sayfasında.", "material_composition": "%98 Pamuk, %2 Elastan", "category": "pantolon"}
    }
}

# Global Kaynaklar (Lifespan veya test fixture'ları tarafından doldurulacak)
active_sessions: Dict[str, Dict[str, Any]] = {}
MODEL_PATH = (BASE_DIR / "nlu_model.bin").resolve()
TURKISH_FREQUENCY_DICTIONARY_PATH = (BASE_DIR / "turkish_frequency_dictionary.txt").resolve()

nlu_model: Optional[fasttext.FastText._FastText] = None
morphology: Optional[zeyrek.MorphAnalyzer] = None
sym_spell: Optional[SymSpell] = None
db_product_original_to_processed_map: Dict[str, str] = {}
CRITICAL_RESOURCES_LOADED = False
LIFESPAN_EXECUTED_FLAG = False

product_extraction_intents = ["fiyat_sorgulama", "ürün_bilgisi_sorma", "stok_sorgulama", "iade_sorgulama", "ürün_malzeme_sorma"]

# Regex Kuralları (İyileştirmeler ve Düzeltmeler)
rules = {
    "calisma_saatleri_sorma": re.compile(
        r"\b((?:çalışma|calisma)\s+saatleri(?:niz)?(?:[\s,.]*nedir\??)?|kaça\s+kadar\s+açık|ne\s+zaman\s+açık|açılış\s+kapanış|mesai|hafta\s*sonu\s+açık|pazar\s+açık\s*mı|hangi\s+saatler|ne\s+zaman\s+kapanıyor|saat\s+kaçta\s+açılıyor|saat\s+kaçta\s+kapanır|açıksınız|calisma\s+saati)\b",
        re.IGNORECASE
    ),
    "kargo_bilgisi_sorma": re.compile(
        r"\b(kargo|gönderim|teslimat|kaç\s+günde\s+gelir|kargo\s+ücret|kargo\s+ne\s+kadar|kargo\s+takip|yurtiçi\s+kargo|kargo\s+nekadar|kargonuz\s+kaç\s+günde|teslim\s+süresi|kargo\s+tutar)\b",
        re.IGNORECASE
    ),
    "fiyat_sorgulama": re.compile(
        r"\b(fiyat|ücret|kaç\s+para|ne\s+kadar|kaç\s+tl|maliyet|ederi|kaça|nekadar|fyt|fiyt|fyaat|fiyay|kça\s+para|ne\s+kadr|fiyatı\s+ne|fiyatı\s+nedir|ücreti\s+nedir|fiyatını\s+öğren|fiyat\s+bilgisi)"
        r"(?!.*(?:kargo|teslimat|gönderim|açık|kapanış|saatler\w*|saat|iade|stok|malzeme|özellik|beden|nerede|adres|konum|telefon|mail|ödeme|site|çalışma|calisma|kumaş|içerik)\b)\b",
        re.IGNORECASE
    ),
    "selamlama": re.compile(
        r"^\s*(merhaba|selam|iyi\s+günler|günaydın|mrb|slm|sa|selamun\s+aleykum|hey|kolay\s+gelsin|merhba|gunaydn|selamlarr|meraba|s\.a\.|nbr|heyo|selamlar|iyi\s+akşamlar|hayırlı\s+işler)\b",
        re.IGNORECASE
    ),
    "iade_sorgulama": re.compile(
        r"\b(iade|geri\s+verme|değişim|değiştir|iade\s+edebilir|iade\s+koşul(?:lar[ıi])?|koşullaeı|para\s+iadesi|değiştirebilir\s+miyim|geri\s+gönderebilir|ürünü\s+geri\s+al|beğenmedim)\b",
        re.IGNORECASE
    ),
    "stok_sorgulama": re.compile(
        r"\b(stokta\s+mevcut\s+mu|stokta\s+var\s*m[ıi]|stok\s+durumu|elde\s+var\s+mı|beden[a-zıüöçşğİ.]*\s+var\s*m[ıi]|bednleri\s+var\s*m[ıi]|numarası\s+var\s+mı|modeli\s+var\s+mı|bulunur\s+mu|kaldı\s+mı|bedeni\s+var\s+mı|stok|bedenleri|mevcutmu)\b"
        r"(?!.*(?:taksit|ödeme|fiyat|malzeme|kumaş|içerik|saat)\b)\b", # "mevcutmu" eklendi, "saat" çıkarıldı
        re.IGNORECASE
    ),
    "tesekkur": re.compile(
        r"^\s*(teşekkür\s+ederim|sağ\s+olun|çok\s+teşekkürler|tşk|eyvallah|sağol|teşekkürler|mersi|saol|eyw|tskler|tesekkurler|teşekürler|saolun|varol)\b",
        re.IGNORECASE
    ),
    "ürün_malzeme_sorma": re.compile(
        r"\b(malzeme|içerik|kumaş|astar|yapılmış|üretilmiş|neyden\s+yapıl|materyal|kumas\s+ne|içeriğinde\s+ne\s+var|kompozisyonu)\b"
        r"(?!.*(?:stok|beden|fiyat|kaç\s+para|ne\s+kadar|bilgi|özellik)\b)\b",
        re.IGNORECASE
    ),
    "ürün_bilgisi_sorma": re.compile(
        r"\b(özellikleri|hakkında\s+bilgi|detay|açıklama|nedir\s+bu|ne\s+işe\s+yarar|urun\s+bilgisi|ürünle\s+ilgili|model\s+hakkında|ürün\s+ne\s+için|anlatır\s+mısın\s+bu\s+ürün|spesifikasyonları)\b"
        r"(?!.*(?:malzeme|kumaş|içerik)\b)\b", # Malzeme ile çakışmayı önle
        re.IGNORECASE
    ),
    "lokasyon_sorma": re.compile(
        r"\b(nerede|adres|konum|yeriniz|mağaza\s+nerede|dükkan\s+nerede|nasıl\s+gel|nerdesiniz|konm|adresiniz\s+neydi|dükkan\s+nerde|hangi\s+semtte|yol\s+tarifi|magazanız)\b",
        re.IGNORECASE
    ),
    "tel_no_sorma": re.compile(
        r"\b(telefon|tel\s+no|numara|iletişim\s+no|arayabilir|whatsapp|mail|e-posta|eposta|numaranız|mail\s+adresiniz|irtibat)\b",
        re.IGNORECASE
    ),
    "odeme_yontemleri_sorma": re.compile(
        r"\b(nasıl\s+öde|ödeme\s+seçenek|ne\s+kabul|kredi\s+kartı|taksit|kapıda\s+ödeme|havale|eft|ödeme\s+türleri|ödeme\s+yapabilir|taksit\s+imkanı|ödeme\s+şekilleri)\b"
        r"(?!.*(?:stok|beden)\b)\b", # "var mı" çıkarıldı, stok/beden ile çakışmayı önle
        re.IGNORECASE
    ),
    "websitesi_sorma": re.compile(r"\b(web\s+site|internet\s+site|online\s+mağaza|link|ürünlere\s+nereden\s+bak|sitenizden\s+sipariş|sayfanız|www|site\s+adres|e-ticaret)\b", re.IGNORECASE),
    "musteri_hizmetlerine_baglanma": re.compile(r"\b(müşteri\s+hizmet|yetkili\s+biri|canlı\s+destek|insanla\s+konuş|temsilciye\s+aktar|operatöre\s+bağlan|birine\s+bağla)\b", re.IGNORECASE),
    "siparis_durumu_sorma": re.compile(r"\b(siparişim\s+ne\s+durumda|kargom\s+nerede|siparişimi\s+takip|kargom\s+ne\s+zaman\s+gelir|sipariş\s+no\s+.*\s+ne\s+oldu|ürünüm\s+gelmedi|kargo\s+gelmedi|sipariş\s+durumu)\b", re.IGNORECASE),
    "oneri_isteme": re.compile(r"\b(ne\s+önerirsin|tavsiye\s+eder|en\s+çok\s+satan|benzer\s+ne\s+var|alternatif\s+ne|öneri\s+var\s+mı|ne\s+tavsiye|bir\s+şey\s+öner|hangi\s+ürünü\s+almalı|ne\s+seçmeli)\b", re.IGNORECASE),
    "olumsuz_yanıt": re.compile(r"^\s*(hayır|yok\s+kalsın|gerek\s+yok|istemiyorum|düşünmüyorum|pas|vazgeçtim|kalsın|olmaz|hayr|ilgilenmiyorum|almayayım)\b", re.IGNORECASE),
}

# Niyet Tespit Parametreleri
GENERAL_INTENTS_FOR_OVERRIDE = ["selamlama", "tesekkur", "olumsuz_yanıt"]
MIN_WORDS_FOR_SLM_OVERRIDE = 2
SLM_OVERRIDE_CONFIDENCE_THRESHOLD = 0.60 # Daha kesin SLM override'ları için artırıldı
FUZZY_MATCH_THRESHOLD = 80 # Ürün eşleştirme için biraz artırıldı

PROTECTED_WORDS_SYMSPELL = {
    "mrb", "slm", "tşk", "eyw", "tmm", "ok", "sa", "kot", "fiyat",
    "s", "m", "l", "xl", "xs", "xxl",
}
SYMSPELL_BLOCKED_CORRECTIONS = {
    ("ceketler", "cesetler"), ("kot", "koy"), ("Kot", "not"),
}
# Bilinen yazım hataları ve doğru karşılıkları (SymSpell sözlüğüne eklenebilir veya burada kullanılabilir)
KNOWN_TYPOS = {
    "pantoon": "pantolon",
    "pntolon": "pantolon",
    "jeket": "ceket",
    "kadr": "kadar",
    "nedr": "nedir",
    "bedn": "beden",
    "bednleri": "bedenleri",
    # "ktene": "keten" # Bu riskli olabilir, SymSpell'in kendi düzeltmesi daha iyi olabilir
}


def _safe_lemmatize_word(word: str) -> str:
    global morphology
    if not morphology: return word.lower()
    word_lower = word.lower()
    if not word_lower.strip() or word_lower.isdigit() or \
       (len(word_lower) <= 2 and word_lower not in ['o', 'bu', 'şu', 'ne', 'mi', 's', 'm', 'l', 'x']):
        return word_lower
    # "fiyat" gibi kelimeler için özel durum
    if word_lower == "fiyat": return "fiyat"

    analysis = morphology.analyze(word_lower)
    if analysis and analysis[0] and analysis[0][0]:
        best_analysis = analysis[0][0]
        lemma = best_analysis.lemma.lower()
        if lemma == "unk" or "Unk" in best_analysis.pos or "Punc" in best_analysis.pos or \
           (len(lemma) < 2 and word_lower != lemma and not word_lower.isdigit() and word_lower not in ['o', 'bu', 'şu', 'ne', 'mi', 's', 'm', 'l', 'x']):
            return word_lower
        return lemma
    return word_lower

def _preprocess_text_for_matching(text_phrase: str) -> str:
    global morphology
    if not text_phrase or not text_phrase.strip(): return ""
    lower_text = text_phrase.lower().strip()
    # Bilinen yazım hatalarını önce düzelt
    for typo, correction in KNOWN_TYPOS.items():
        lower_text = lower_text.replace(typo, correction)

    cleaned_text = re.sub(r"[^\w\s'-]", " ", lower_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    words = cleaned_text.split()
    lemmatized_words = [_safe_lemmatize_word(word) for word in words if word]
    final_text = " ".join(lemmatized_words).strip()
    return final_text

def correct_spelling(text: str) -> str:
    global sym_spell, PROTECTED_WORDS_SYMSPELL, SYMSPELL_BLOCKED_CORRECTIONS, KNOWN_TYPOS
    if not sym_spell:
        logger.warning("SymSpell (correct_spelling): Model is None. Spell check skipped.")
        # SymSpell yoksa bile bilinen typo'ları düzeltmeyi dene
        temp_text = text.lower()
        for typo, correction in KNOWN_TYPOS.items():
            temp_text = temp_text.replace(typo, correction)
        if temp_text != text.lower():
            logger.info(f"Known typo correction (SymSpell None): '{text}' -> '{temp_text}'")
            return temp_text # Case'i korumak için orijinal text'in case'ini uygulamak gerekebilir
        return text

    if not text or not text.strip(): return text

    # Önce bilinen typo'ları düzelt
    temp_corrected_text = text
    for typo, correction in KNOWN_TYPOS.items():
        # Kelime bazlı replace daha güvenli olabilir
        temp_corrected_text = re.sub(r'\b' + re.escape(typo) + r'\b', correction, temp_corrected_text, flags=re.IGNORECASE)

    words = temp_corrected_text.split()
    corrected_words = []
    text_changed_overall = (temp_corrected_text != text)


    known_size_abbr = {"S", "M", "L", "XL", "XS", "XXL"}
    numeric_size_pattern = re.compile(r"^\d{2,3}$")
    # "mı" gibi ekleri ayırmadan önce bedenleri yakala
    size_with_optional_suffix_pattern = re.compile(r"^(\d{2,3}|[SMLX]{1,3})(?:[siıuüö]{1,2})?$", re.IGNORECASE)


    for word_val in words:
        original_word_lower = word_val.lower()
        corrected_word_to_append = word_val

        if original_word_lower in PROTECTED_WORDS_SYMSPELL:
            corrected_words.append(word_val)
            continue

        # Bedenleri ve sayısal bedenleri öncelikli olarak koru
        size_suffix_match = size_with_optional_suffix_pattern.match(word_val)
        if size_suffix_match:
            core_part = size_suffix_match.group(1).upper()
            if core_part.isdigit() and (28 <= int(core_part) <= 60):
                corrected_words.append(core_part)
                if core_part != word_val: text_changed_overall = True
                continue
            elif core_part in known_size_abbr:
                corrected_words.append(core_part)
                if core_part != word_val: text_changed_overall = True
                continue

        suggestions = sym_spell.lookup(word_val, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)

        if suggestions:
            best_suggestion_obj = suggestions[0]
            best_suggestion_term = best_suggestion_obj.term

            if (original_word_lower, best_suggestion_term.lower()) in SYMSPELL_BLOCKED_CORRECTIONS:
                logger.warning(f"SymSpell: Explicitly BLOCKED correction from '{word_val}' to '{best_suggestion_term}'. Keeping original.")
                corrected_words.append(word_val)
                continue

            if best_suggestion_term.lower() != original_word_lower and best_suggestion_obj.distance > 0:
                is_safe_correction = True
                if len(word_val) > 3 and word_val[0].lower() != best_suggestion_term[0].lower() and \
                   (word_val.lower() not in best_suggestion_term.lower() and best_suggestion_term.lower() not in word_val.lower()):
                    # Eğer ilk harf değişiyorsa ve alt dize değilse, daha dikkatli ol
                    # Örn: "kot" -> "koy" (distance 1). Bunu engellemek için sözlükte "kot" olmalı.
                    # Eğer "kot" sözlükte yoksa ve "koy" varsa, bu düzeltme yapılabilir.
                    # Bu heuristik, sözlüğün kalitesine çok bağlı.
                    if fuzz.ratio(original_word_lower, best_suggestion_term.lower()) < 70: # Daha düşük benzerlikteyse riskli say
                        logger.warning(f"SymSpell: Risky correction (first letter changed, low similarity): '{word_val}' -> '{best_suggestion_term}'. Keeping original.")
                        is_safe_correction = False

                if is_safe_correction:
                    logger.info(f"SymSpell Correction Applied: '{word_val}' -> '{best_suggestion_term}'")
                    corrected_word_to_append = best_suggestion_term
                    text_changed_overall = True
            elif best_suggestion_term != word_val :
                logger.info(f"SymSpell: Minor correction (case or zero-distance): '{word_val}' -> '{best_suggestion_term}'")
                corrected_word_to_append = best_suggestion_term
                text_changed_overall = True
        corrected_words.append(corrected_word_to_append)

    if text_changed_overall:
        corrected_phrase = " ".join(corrected_words)
        logger.info(f"Spell Correction Result: Original='{text}' -> Corrected='{corrected_phrase}'")
        return corrected_phrase

    logger.debug(f"No significant spell correction for: '{text}'")
    return text


def extract_simple_entities(original_query_spell_checked: str, processed_query_lemmatized: str, intent: Optional[str] = None) -> Dict[str, Any]:
    global morphology, db_product_original_to_processed_map

    current_morphology = morphology
    current_db_map = db_product_original_to_processed_map

    product_candidate_original_name: Optional[str] = None
    logger.debug(f"--- Starting Entity Extraction ---")
    logger.debug(f"Input original_query_spell_checked: '{original_query_spell_checked}'")
    logger.debug(f"Input processed_query_lemmatized: '{processed_query_lemmatized}'")
    logger.debug(f"Input intent: '{intent}'")
    logger.debug(f"Entity Extraction - Morphology type: {type(current_morphology)}")
    logger.debug(f"Entity Extraction - DB Map size: {len(current_db_map if current_db_map else {})}")


    entities = {"product": None, "size": None, "is_generic_product_term": False, "generic_term_options": []}

    generic_product_terms_map: Dict[str, List[str]] = {}
    lemmatized_generic_categories: set[str] = set()
    if current_morphology:
        for orig_name_from_db, prod_details in simulated_business_data["products"].items():
            category = prod_details.get("category")
            if category:
                lem_category = _preprocess_text_for_matching(category)
                if lem_category:
                    lemmatized_generic_categories.add(lem_category)
                    if lem_category not in generic_product_terms_map:
                        generic_product_terms_map[lem_category] = []
                    generic_product_terms_map[lem_category].append(orig_name_from_db)
        logger.debug(f"Initialized Lemmatized Generic Categories: {lemmatized_generic_categories}")
    else:
        logger.warning("Morphology not loaded in extract_simple_entities, generic category processing will be impaired.")

    # --- Beden Çıkarımı ---
    # "mı" gibi eklerden etkilenmemesi için daha dikkatli bir regex
    # Önce "var mı" gibi ifadeleri temizleyebiliriz veya regex'i daha spesifik yapabiliriz.
    query_for_size = original_query_spell_checked
    # " var m" veya " var mi" gibi ifadeleri geçici olarak değiştirerek "m" harfinin beden olarak algılanmasını engelle
    query_for_size_cleaned = re.sub(r"\bvar\s+m[ıi]?\b", " var_soru ", query_for_size, flags=re.IGNORECASE)

    logger.debug(f"Size Extraction: Searching for size in (cleaned): '{query_for_size_cleaned}'")
    # "smal" gibi yazım hatalarını da içeren daha geniş bir beden paterni
    size_pattern = r"\b(\d{2,3}|(?:X?S|M|L|X{1,2}L)|small|medium|large|xsmall|xlarge|xxlarge|ekstra\s+large|x\s+large|sml|smal)\b" # beden kelimesi olmadan
    size_match = re.search(size_pattern, query_for_size_cleaned, re.IGNORECASE)

    if size_match:
        extracted_size_token = size_match.group(1).lower()
        # core_size_token = re.sub(r"[\s-]*beden[a-zıüöçşğİ.]*$", "", extracted_size_token).strip().upper()
        core_size_token = extracted_size_token.upper() # Zaten beden kelimesi yok

        size_normalization_map = {
            "SMALL": "S", "MEDIUM": "M", "LARGE": "L", "XSMALL": "XS",
            "XLARGE": "XL", "XXLARGE": "XXL", "EKSTRA LARGE": "XL", "X LARGE": "XL",
            "SML": "S", "SMAL": "S"
        }
        if core_size_token.isdigit():
            try:
                size_num = int(core_size_token)
                if 28 <= size_num <= 60: entities["size"] = core_size_token
            except ValueError: pass
        elif core_size_token in size_normalization_map:
            entities["size"] = size_normalization_map[core_size_token]
        elif core_size_token in ["S", "M", "L", "XL", "XXL", "XS"]:
             entities["size"] = core_size_token
        if entities.get("size"): logger.info(f"Size extracted and normalized: '{entities['size']}'")
    else: logger.info(f"No size match found in '{query_for_size_cleaned}'")


    # --- Ürün Çıkarımı ---
    anaphoric_pronouns = {"bu", "şu", "o", "bunun", "şunun", "onun"}
    query_words_for_product = processed_query_lemmatized.split()
    is_likely_anaphoric = False
    if query_words_for_product and query_words_for_product[0] in anaphoric_pronouns:
        potential_product_phrase = " ".join(query_words_for_product[1:])
        if not current_db_map or (not any(proc_db_name in potential_product_phrase for proc_db_name in current_db_map.values()) and \
           not any(lem_cat in potential_product_phrase for lem_cat in lemmatized_generic_categories)):
            is_likely_anaphoric = True
            logger.info(f"Marked as anaphoric: '{processed_query_lemmatized}'")

    if is_likely_anaphoric and intent in product_extraction_intents:
        logger.info(f"Product extraction skipped for anaphoric query.")
    else:
        logger.debug(f"Product Extraction: Searching in (lemmatized): '{processed_query_lemmatized}'")
        processed_db_names_to_original_map = {v: k for k, v in (current_db_map or {}).items()}

        # 1. En Uzun Tam Eşleşme (Lemmatize Edilmiş DB İsimleriyle)
        # Bu, "kot pantolon" gibi daha spesifik isimlerin "pantolon" gibi genel terimlerden önce eşleşmesine yardımcı olur.
        if processed_db_names_to_original_map:
            # Önce tam çok kelimeli ürün adlarını ara
            # Örneğin, "kot pantolon" (lemmatize edilmiş) -> "Kot Pantolon" (orijinal)
            # "ipek gömlek" (lemmatize edilmiş) -> "İpek Gömlek" (orijinal)
            # current_db_map: {'Keten Pantolon': 'keten pantolon', 'İpek Gömlek': 'ipek gömlek', ...}
            # processed_query_lemmatized: 'ipek gömlek small be'
            # processed_query_lemmatized: 'kot pantolon 30 be var mı'

            # En uzun eşleşmeyi bulmak için
            best_exact_match_original_name = None
            best_exact_match_len = 0

            for original_db_name, processed_db_name in (current_db_map or {}).items():
                if processed_db_name and re.search(r"\b" + re.escape(processed_db_name) + r"\b", processed_query_lemmatized):
                    if len(processed_db_name) > best_exact_match_len:
                        best_exact_match_len = len(processed_db_name)
                        best_exact_match_original_name = original_db_name

            if best_exact_match_original_name:
                product_candidate_original_name = best_exact_match_original_name
                logger.info(f"Product Longest Exact Match (Lemmatized DB): '{product_candidate_original_name}' in query '{processed_query_lemmatized}'")


        # 2. Genel Kategori Eşleşmesi (Eğer spesifik ürün bulunamadıysa)
        if not product_candidate_original_name and lemmatized_generic_categories:
            found_generic_term_key = None
            # En uzun genel kategoriyi bul
            for lem_cat in sorted(list(lemmatized_generic_categories), key=len, reverse=True):
                if lem_cat and re.search(r"\b" + re.escape(lem_cat) + r"\b", processed_query_lemmatized):
                    found_generic_term_key = lem_cat
                    logger.info(f"Generic Category Match (Regex in lemmatized query): '{found_generic_term_key}'")
                    break
            if found_generic_term_key:
                entities["is_generic_product_term"] = True
                entities["product"] = found_generic_term_key # Lemmatize kategori adı
                entities["generic_term_options"] = generic_product_terms_map.get(found_generic_term_key, [])


        # 3. Bulanık Eşleşme (Eğer hala spesifik ürün bulunamadıysa VE generic de değilse)
        if not product_candidate_original_name and not entities.get("is_generic_product_term"):
            if processed_db_names_to_original_map and processed_query_lemmatized:
                # RapidFuzz için sorgudan sadece ürünle ilgili olabilecek kısmı almayı dene
                # Örneğin "Keten pntolon fiyatı" -> "keten pntolon" (lemmatize edilmiş hali)
                # Bu, `extract_simple_entities` fonksiyonunun girdisi olan `processed_query_lemmatized` üzerinde yapılmalı.
                # Anahtar kelimeleri (fiyat, beden vb.) çıkar.
                product_phrase_for_fuzz = processed_query_lemmatized
                trailing_keywords = ["fiyat", "beden", "stok", "var", "mı", "ne", "kaç", "kadar", "özellik", "malzeme", "içerik", "kumaş", "nedir", "acaba", "sor"]
                temp_words = product_phrase_for_fuzz.split()
                # Sondan başlayarak anahtar kelimeleri çıkar
                while temp_words and temp_words[-1] in trailing_keywords:
                    temp_words.pop()
                # Baştan başlayarak anahtar kelimeleri çıkar (daha az yaygın ama olabilir)
                while temp_words and temp_words[0] in trailing_keywords:
                    temp_words.pop(0)
                product_phrase_for_fuzz = " ".join(temp_words).strip()

                if product_phrase_for_fuzz:
                    logger.debug(f"RapidFuzz: Using phrase '{product_phrase_for_fuzz}' for matching.")
                    choices = list(processed_db_names_to_original_map.keys()) # İşlenmiş DB adları
                    best_match_tuple = process.extractOne(product_phrase_for_fuzz, choices, scorer=fuzz.token_set_ratio, score_cutoff=FUZZY_MATCH_THRESHOLD)
                    if best_match_tuple:
                        matched_processed_db_name, score, _ = best_match_tuple
                        product_candidate_original_name = processed_db_names_to_original_map[matched_processed_db_name]
                        logger.info(f"Product RapidFuzz Match: '{product_candidate_original_name}' (Score: {score:.2f}) for phrase '{product_phrase_for_fuzz}'")
                    else:
                        logger.info(f"RapidFuzz: No specific product match for phrase '{product_phrase_for_fuzz}'.")
                else:
                    logger.info(f"RapidFuzz: Phrase became empty after stripping keywords from '{processed_query_lemmatized}'. Skipping.")

    if not entities.get("is_generic_product_term") and product_candidate_original_name:
        entities["product"] = product_candidate_original_name

    logger.info(f"--- Ending Entity Extraction --- Entities: {entities}")
    return entities


def call_slm_model(processed_query_for_slm: str) -> NLUSingleAnalysis:
    global nlu_model
    current_nlu_model = nlu_model

    if not current_nlu_model:
        logger.error("SLM call failed: NLU model (global) is None.")
        return NLUSingleAnalysis(slm_intent="slm_modeli_yüklenemedi", slm_entities=[], confidence_score=0.0, message="SLM (fastText) model (global) is None.")

    if not processed_query_for_slm or not processed_query_for_slm.strip() or len(processed_query_for_slm.strip()) < 2 :
        logger.warning(f"SLM call skipped: Processed query for SLM is too short or empty: '{processed_query_for_slm}'")
        return NLUSingleAnalysis(slm_intent="tahmin_yok_slm_ile_kısa_sorgu", slm_entities=[], confidence_score=0.0, message="Query too short for SLM.")

    cleaned_query_for_slm = processed_query_for_slm.replace("\n", " ")
    logger.debug(f"Query sent to SLM model: '{cleaned_query_for_slm}'")
    predictions = current_nlu_model.predict(cleaned_query_for_slm, k=1)

    intent_name = "tahmin_yok_slm_ile"; confidence = 0.0
    if predictions and predictions[0] and predictions[1] and predictions[0][0] and predictions[1][0]:
        predicted_label_full = predictions[0][0]; confidence = predictions[1][0]
        intent_name = predicted_label_full.replace("__label__", "")
    logger.info(f"SLM MODEL CALL: Query='{processed_query_for_slm}' -> Intent: {intent_name}, Confidence: {confidence:.4f}")
    return NLUSingleAnalysis(slm_intent=intent_name, slm_entities=[], confidence_score=float(f"{confidence:.4f}"), message="Response from fastText SLM model.")

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global morphology, nlu_model, sym_spell, db_product_original_to_processed_map
    global CRITICAL_RESOURCES_LOADED, LIFESPAN_EXECUTED_FLAG

    logger.info("Lifespan: Application startup sequence initiated...")
    app_instance.state.lifespan_was_executed = False
    app_instance.state.critical_resources_loaded = False
    app_instance.state.morphology = None
    app_instance.state.nlu_model = None
    app_instance.state.sym_spell = None
    app_instance.state.db_product_original_to_processed_map = {}

    zeyrek_loaded, fasttext_loaded, symspell_loaded = False, False, False
    try:
        logger.info("LIFESPAN: Loading Zeyrek MorphAnalyzer...")
        morphology = zeyrek.MorphAnalyzer()
        app_instance.state.morphology = morphology
        logger.info(f"LIFESPAN: Zeyrek loaded. Type: {type(morphology)}")
        zeyrek_loaded = True
    except Exception as e: logger.error("LIFESPAN: CRITICAL ERROR loading Zeyrek: %s", e, exc_info=True)

    if MODEL_PATH.exists():
        try:
            logger.info(f"LIFESPAN: Loading NLU model from {MODEL_PATH}...")
            nlu_model = fasttext.load_model(str(MODEL_PATH))
            app_instance.state.nlu_model = nlu_model
            logger.info(f"LIFESPAN: NLU model loaded. Type: {type(nlu_model)}")
            fasttext_loaded = True
        except Exception as e: logger.error("LIFESPAN: CRITICAL ERROR loading NLU model: %s", e, exc_info=True)
    else: logger.error(f"LIFESPAN: NLU model file NOT FOUND at {MODEL_PATH}")

    if TURKISH_FREQUENCY_DICTIONARY_PATH.exists():
        try:
            logger.info(f"LIFESPAN: Loading SymSpell dictionary from {TURKISH_FREQUENCY_DICTIONARY_PATH}...")
            temp_sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            # Sözlüğe bilinen typo'ları ekle
            for typo, correction in KNOWN_TYPOS.items():
                temp_sym_spell.create_dictionary_entry(correction, temp_sym_spell.get_count(correction) or 1) # Düzeltmenin frekansını al veya 1 yap
                temp_sym_spell.create_dictionary_entry(typo, 0) # Typo'yu düşük frekansla ekle

            if temp_sym_spell.load_dictionary(str(TURKISH_FREQUENCY_DICTIONARY_PATH), term_index=0, count_index=1, separator="\t", encoding="utf-8"):
                sym_spell = temp_sym_spell
                app_instance.state.sym_spell = sym_spell
                logger.info(f"LIFESPAN: SymSpell dictionary loaded. Word count: {len(sym_spell.words) if sym_spell else 0}")
                symspell_loaded = True
            else: logger.error(f"LIFESPAN: SymSpell dictionary load_dictionary returned False for {TURKISH_FREQUENCY_DICTIONARY_PATH}")
        except Exception as e: logger.error("LIFESPAN: ERROR loading SymSpell: %s", e, exc_info=True)
    else: logger.warning(f"LIFESPAN: SymSpell dictionary NOT FOUND at {TURKISH_FREQUENCY_DICTIONARY_PATH}. Spell checking impaired.")

    CRITICAL_RESOURCES_LOADED = zeyrek_loaded and fasttext_loaded
    LIFESPAN_EXECUTED_FLAG = True
    app_instance.state.critical_resources_loaded = CRITICAL_RESOURCES_LOADED
    app_instance.state.lifespan_was_executed = LIFESPAN_EXECUTED_FLAG

    logger.info(f"LIFESPAN Summary: CriticalLoaded={CRITICAL_RESOURCES_LOADED}, SymSpellLoaded={symspell_loaded}")

    if morphology:
        temp_db_map = {}
        for original_name, details in simulated_business_data["products"].items():
            temp_db_map[original_name] = _preprocess_text_for_matching(original_name)
        db_product_original_to_processed_map.clear() # Önce temizle
        db_product_original_to_processed_map.update(temp_db_map) # Sonra güncelle
        app_instance.state.db_product_original_to_processed_map = db_product_original_to_processed_map.copy()
        logger.info(f"LIFESPAN: DB product map processed. Size: {len(db_product_original_to_processed_map)}")
        logger.debug(f"DB Product Map Content (Lifespan): {db_product_original_to_processed_map}")

    logger.info("--- LIFESPAN --- YIELDING TO APPLICATION ---")
    yield
    logger.info("--- LIFESPAN --- SHUTDOWN SEQUENCE ---")
    morphology = None; nlu_model = None; sym_spell = None; db_product_original_to_processed_map.clear()
    CRITICAL_RESOURCES_LOADED = False; LIFESPAN_EXECUTED_FLAG = False
    app_instance.state.morphology = None; app_instance.state.nlu_model = None; app_instance.state.sym_spell = None
    app_instance.state.db_product_original_to_processed_map = {}
    app_instance.state.critical_resources_loaded = False; app_instance.state.lifespan_was_executed = False
    logger.info("--- LIFESPAN --- SHUTDOWN COMPLETE ---")

app = FastAPI(lifespan=lifespan)

@app.post("/process_query/", response_model=NLUResponse)
async def process_query(payload: QueryRequest, request: FastAPIRequest):
    original_user_query = payload.query
    logger.info(f"--- Received Query: '{original_user_query}', Session ID: {payload.session_id} ---")

    if not LIFESPAN_EXECUTED_FLAG or not CRITICAL_RESOURCES_LOADED:
        error_msg = "Sistem hatası: Temel NLU kaynakları yüklenemedi."
        logger.error(f"CRITICAL process_query: {error_msg} (LifespanRan={LIFESPAN_EXECUTED_FLAG}, CriticalLoaded={CRITICAL_RESOURCES_LOADED})")
        raise HTTPException(status_code=503, detail=error_msg)

    if sym_spell is None:
         logger.warning("SymSpell (process_query): Model is None. Spell checking will be impaired.")

    effective_session_id = payload.session_id or str(uuid.uuid4())
    if payload.session_id is None: logger.info(f"New session_id created: {effective_session_id}")

    if effective_session_id not in active_sessions:
        active_sessions[effective_session_id] = {"history": [], "last_updated": datetime.now(), "last_mentioned_product": None}
    else:
        active_sessions[effective_session_id]["last_updated"] = datetime.now()
    previous_query_from_session = active_sessions[effective_session_id]["history"][-1]["query"] if active_sessions[effective_session_id]["history"] else None

    if not original_user_query or not original_user_query.strip():
        raise HTTPException(status_code=400, detail="Sorgu boş olamaz.")

    active_sessions[effective_session_id]["history"].append({"query": original_user_query, "timestamp": datetime.now()})
    active_sessions[effective_session_id]["history"] = active_sessions[effective_session_id]["history"][-5:]

    query_for_regex_detection = correct_spelling(original_user_query.lower())
    processed_query_for_nlu_pipeline = preprocess_query_for_nlu(original_user_query)

    nlu_method = "initial_regex_search"; final_intent = None; slm_analysis_result: Optional[NLUSingleAnalysis] = None
    detected_intent_via_regex = None
    intent_priority_order = [
        "selamlama", "tesekkur", "olumsuz_yanıt", "calisma_saatleri_sorma", "kargo_bilgisi_sorma",
        "lokasyon_sorma", "tel_no_sorma", "odeme_yontemleri_sorma", "websitesi_sorma", "iade_sorgulama",
        "fiyat_sorgulama", "stok_sorgulama", "ürün_malzeme_sorma", "ürün_bilgisi_sorma",
        "musteri_hizmetlerine_baglanma", "siparis_durumu_sorma", "oneri_isteme"
    ]

    logger.debug(f"Intent Detection: Query for regex (spell-checked lower): '{query_for_regex_detection}'")
    for intent_key in intent_priority_order:
        pattern = rules.get(intent_key)
        if not pattern: continue
        match_condition = pattern.match if intent_key in GENERAL_INTENTS_FOR_OVERRIDE else pattern.search
        if match_condition(query_for_regex_detection):
            if intent_key == "fiyat_sorgulama" and rules["calisma_saatleri_sorma"].search(query_for_regex_detection):
                logger.debug(f"Regex: '{intent_key}' matched but also 'calisma_saatleri_sorma'. Skipping '{intent_key}'.")
                continue
            if intent_key == "stok_sorgulama" and rules["odeme_yontemleri_sorma"].search(query_for_regex_detection):
                logger.debug(f"Regex: '{intent_key}' matched but also 'odeme_yontemleri_sorma'. Skipping '{intent_key}'.")
                continue
            # "Deri Ceket XL kaldı mı" -> tesekkur olmaması için kontrol
            if intent_key == "tesekkur" and rules["stok_sorgulama"].search(query_for_regex_detection):
                logger.debug(f"Regex: '{intent_key}' matched but also 'stok_sorgulama' (e.g., 'kaldı mı'). Skipping '{intent_key}'.")
                continue


            detected_intent_via_regex = intent_key
            logger.info(f"Intent Detection: Regex initial match: '{detected_intent_via_regex}'")
            break

    query_for_slm = processed_query_for_nlu_pipeline
    if detected_intent_via_regex in GENERAL_INTENTS_FOR_OVERRIDE and len(original_user_query.split()) >= MIN_WORDS_FOR_SLM_OVERRIDE:
        logger.info(f"Intent Logic: Regex general intent ('{detected_intent_via_regex}'). Consulting SLM. SLM query: '{query_for_slm}'.")
        potential_slm_analysis = call_slm_model(query_for_slm)
        if potential_slm_analysis.slm_intent not in GENERAL_INTENTS_FOR_OVERRIDE and \
           potential_slm_analysis.slm_intent not in ["kapsam_disi", "tahmin_yok_slm_ile", "slm_modeli_yüklenemedi", "tahmin_yok_slm_ile_kısa_sorgu"] and \
           potential_slm_analysis.confidence_score >= SLM_OVERRIDE_CONFIDENCE_THRESHOLD:
            final_intent = potential_slm_analysis.slm_intent
            slm_analysis_result = potential_slm_analysis
            nlu_method = f"slm_override_of_{detected_intent_via_regex}"
        else:
            final_intent = detected_intent_via_regex
            nlu_method = f"regex_kept_{detected_intent_via_regex}"
    elif detected_intent_via_regex is not None:
        final_intent = detected_intent_via_regex
        nlu_method = f"regex_specific_{final_intent}"
    else:
        logger.info(f"Intent Logic: No regex match. Consulting SLM. SLM query: '{query_for_slm}'")
        slm_analysis_result = call_slm_model(query_for_slm)
        final_intent = slm_analysis_result.slm_intent
        nlu_method = f"slm_direct_{final_intent}"
        if final_intent in ["tahmin_yok_slm_ile", "slm_modeli_yüklenemedi", "tahmin_yok_slm_ile_kısa_sorgu"] or \
           (final_intent == "kapsam_disi" and slm_analysis_result.confidence_score < 0.5):
            logger.warning(f"SLM result '{final_intent}' (Conf: {slm_analysis_result.confidence_score if slm_analysis_result else 'N/A'}) not reliable. Defaulting to 'kapsam_disi'.")
            final_intent = "kapsam_disi"
            nlu_method += "_fallback_to_kapsam_disi"

    logger.info(f"--- Final Intent: {final_intent}, Method: {nlu_method} ---")

    current_entities = extract_simple_entities(
        original_query_spell_checked=query_for_regex_detection, # SymSpell'den geçmiş hali
        processed_query_lemmatized=processed_query_for_nlu_pipeline, # SymSpell + Lemmatize
        intent=final_intent
    )
    extracted_product_token = current_entities.get("product")
    is_generic_product_term = current_entities.get("is_generic_product_term", False)
    clarification_options_for_generic = current_entities.get("generic_term_options", [])
    resolved_size_entity = current_entities.get("size")

    logger.info(f"Entities Extracted: ProductToken='{extracted_product_token}', Size='{resolved_size_entity}', IsGeneric='{is_generic_product_term}'")

    session_data = active_sessions[effective_session_id]
    last_mentioned_product_from_context = session_data.get("last_mentioned_product")
    resolved_product_for_response: Optional[str] = None

    if extracted_product_token:
        if not is_generic_product_term:
            resolved_product_for_response = extracted_product_token
            session_data["last_mentioned_product"] = resolved_product_for_response
            logger.info(f"Context Update: Specific product '{resolved_product_for_response}' set as last_mentioned_product.")
        else:
            logger.info(f"Context Info: Generic product term (lemmatized) '{extracted_product_token}' found.")
    elif final_intent in product_extraction_intents and last_mentioned_product_from_context:
        # Eğer sorguda ürün yoksa ama niyet ürünle ilgiliyse ve bağlamda ürün varsa onu kullan
        if not extracted_product_token and not is_generic_product_term:
             resolved_product_for_response = last_mentioned_product_from_context
             logger.info(f"Context Usage: Product '{resolved_product_for_response}' resolved from session for intent '{final_intent}'.")


    bot_response_text = f"Üzgünüm, '{original_user_query}' isteğinizi tam anlayamadım."
    actionable_message = f"Intent: {final_intent or 'belirlenemedi'}."
    if resolved_product_for_response: actionable_message += f" Product: {resolved_product_for_response.title()}."
    elif is_generic_product_term and extracted_product_token: actionable_message += f" GenericTerm: {extracted_product_token.title()}."
    if resolved_size_entity: actionable_message += f" Size: {resolved_size_entity}."

    ask_for_clarification_flag = False
    clarification_options_list: Optional[List[str]] = None

    if is_generic_product_term and extracted_product_token and final_intent in product_extraction_intents:
        display_generic_term = extracted_product_token.title()
        if clarification_options_for_generic:
            options_str = ", ".join([f"'{p.title()}'" for p in clarification_options_for_generic])
            bot_response_text = f"'{display_generic_term}' ile ilgili birkaç ürünümüz var: {options_str}. Hangisini sormuştunuz?"
            actionable_message += f" Clarification needed. Options: {options_str}"
            ask_for_clarification_flag = True; clarification_options_list = clarification_options_for_generic
        else:
            bot_response_text = f"'{display_generic_term}' ile ilgili bir ürünümüz bulunmuyor."
    # "bu pantolonun kumaşı nedir" -> ask_for_clarification=True oluyordu, bunu düzeltelim
    elif final_intent in product_extraction_intents and not resolved_product_for_response and not (is_generic_product_term and extracted_product_token):
        if final_intent == "fiyat_sorgulama": bot_response_text = "Hangi ürünün fiyatını soruyorsunuz?"
        elif final_intent == "stok_sorgulama":
            bot_response_text = "Hangi ürünün stok durumunu soruyorsunuz?"
            if resolved_size_entity: bot_response_text += f" ({resolved_size_entity} beden için)"
        elif final_intent == "ürün_malzeme_sorma": bot_response_text = "Hangi ürünün malzeme/içerik bilgisini soruyorsunuz?"
        elif final_intent == "ürün_bilgisi_sorma": bot_response_text = "Hangi ürün hakkında bilgi almak istiyorsunuz?"
        else: bot_response_text = "Lütfen hangi üründen bahsettiğinizi belirtin."
        # Bu durumda netleştirme sormuyoruz, direkt ürün istiyoruz.
        ask_for_clarification_flag = False # Testteki "bu pantolonun kumaşı nedir" için False olmalı
    elif final_intent == "fiyat_sorgulama":
        # resolved_product_for_response burada dolu olmalı (ya direkt çıkarımdan ya da bağlamdan)
        prod_data = simulated_business_data["products"].get(resolved_product_for_response)
        if prod_data: bot_response_text = f"'{resolved_product_for_response.title()}' ürününün fiyatı: {prod_data.get('price', 'Bilinmiyor')}. Detaylar: {prod_data.get('link', '#')}"
        else: bot_response_text = f"'{resolved_product_for_response.title()}' ürününü sistemde bulamadım."
    elif final_intent == "stok_sorgulama":
        prod_data = simulated_business_data["products"].get(resolved_product_for_response)
        if prod_data:
            name = resolved_product_for_response.title()
            sizes = prod_data.get("available_sizes_info", "Beden bilgisi ürün sayfasındadır.")
            link = prod_data.get("link", "#")
            if resolved_size_entity:
                bot_response_text = f"'{name}' {resolved_size_entity} bedeni için stok ve diğer detaylar ürün sayfasındadır: {link}. Genel bedenler: {sizes}."
            else: # Testteki "İpek Gömlek stokta varmi" için "ürününün" yerine "için" kullanıldı
                bot_response_text = f"'{name}' için tüm beden ve stok bilgileri ürün sayfasındadır: {link}. Genel bedenler: {sizes}."
        else: bot_response_text = f"'{resolved_product_for_response.title() if resolved_product_for_response else 'Belirtilmeyen ürün'}' ürününü sistemde bulamadım."
    elif final_intent == "ürün_malzeme_sorma":
        prod_data = simulated_business_data["products"].get(resolved_product_for_response)
        if prod_data: bot_response_text = f"'{resolved_product_for_response.title()}' ürününün malzeme içeriği: {prod_data.get('material_composition', 'Belirtilmemiş')}. Detaylar: {prod_data.get('link', '#')}"
        else: bot_response_text = f"'{resolved_product_for_response.title() if resolved_product_for_response else 'Belirtilmeyen ürün'}' ürününü bulamadım."
    elif final_intent == "ürün_bilgisi_sorma":
        prod_data = simulated_business_data["products"].get(resolved_product_for_response)
        if prod_data: bot_response_text = f"'{resolved_product_for_response.title()}': {prod_data.get('description', 'Açıklama yok.')} Detaylar: {prod_data.get('link', '#')}"
        else: bot_response_text = f"'{resolved_product_for_response.title() if resolved_product_for_response else 'Belirtilmeyen ürün'}' ürününü bulamadım."
    elif final_intent == "iade_sorgulama":
        policy = simulated_business_data["business_info"]["return_policy"]
        if resolved_product_for_response: bot_response_text = f"'{resolved_product_for_response.title()}' ile ilgili iade politikamız: {policy}"
        else: bot_response_text = f"Genel iade politikamız: {policy}"
    elif final_intent == "lokasyon_sorma":
        bot_response_text = f"Adresimiz: {simulated_business_data['business_info']['address']}. Harita: {simulated_business_data['business_info']['maps_link']}"
    elif final_intent == "tel_no_sorma":
        info = simulated_business_data['business_info']
        bot_response_text = f"Telefon: {info['phone']}. WhatsApp: {info['whatsapp_number']}. E-posta: {info['email']}."
    elif final_intent == "calisma_saatleri_sorma":
        bot_response_text = f"Çalışma saatlerimiz: {simulated_business_data['business_info']['opening_hours']}."
    elif final_intent == "odeme_yontemleri_sorma":
        bot_response_text = f"Kabul ettiğimiz ödeme yöntemleri: {simulated_business_data['business_info']['payment_options']}."
    elif final_intent == "kargo_bilgisi_sorma":
        bot_response_text = simulated_business_data['business_info']['shipping_info']
    elif final_intent == "selamlama":
        bot_response_text = "Merhaba! Size nasıl yardımcı olabilirim?"
    elif final_intent == "tesekkur":
        bot_response_text = "Rica ederim! Başka bir konuda yardımcı olabilir miyim?"
    elif final_intent == "olumsuz_yanıt":
        bot_response_text = "Anladım. Başka bir konuda yardımcı olabilir miyim?"
    elif final_intent == "kapsam_disi":
        bot_response_text = "Üzgünüm, bu konuda yardımcı olamıyorum. Ürünlerimiz, fiyatlarımız veya mağazamızla ilgili soru sorabilirsiniz."
    elif final_intent in ["tahmin_yok_slm_ile", "tahmin_yok_slm_ile_kısa_sorgu", "slm_modeli_yüklenemedi"]:
        bot_response_text = "Ne demek istediğinizi tam anlayamadım. Farklı kelimelerle tekrar sorabilir misiniz?"

    logger.info(f"Final Bot Response: '{bot_response_text}'")
    logger.info(f"--- Query Processing End for Session ID: {effective_session_id} ---")

    return NLUResponse(
        original_query=original_user_query,
        processed_query_for_nlu=processed_query_for_nlu_pipeline,
        session_id=effective_session_id,
        nlu_method=nlu_method,
        analysis=slm_analysis_result,
        detected_intent=final_intent,
        previous_query_in_session=previous_query_from_session,
        resolved_product=resolved_product_for_response,
        resolved_size=resolved_size_entity,
        actionable_message=actionable_message,
        bot_response=bot_response_text,
        ask_for_clarification=ask_for_clarification_flag,
        clarification_options=clarification_options_list
    )

@app.get("/")
async def read_root(request: FastAPIRequest):
    api_version_message = "NLU API (v23.8 - SymSpell Guardrails & Enhanced Entity Logging)"

    lifespan_run = LIFESPAN_EXECUTED_FLAG
    critical_loaded = CRITICAL_RESOURCES_LOADED
    symspell_obj = sym_spell
    symspell_loaded_flag = symspell_obj is not None

    status_detail = ""
    if lifespan_run is True:
        if critical_loaded is True:
            status_detail = " - Status: Active, Core NLU Resources (Zeyrek, FastText) Loaded."
            if symspell_loaded_flag:
                status_detail += f" SymSpell Active (Word Count: {len(symspell_obj.words) if hasattr(symspell_obj, 'words') else 'N/A'})."
            else: status_detail += " SymSpell INACTIVE or FAILED to load. Spell checking impaired."
        else: status_detail = " - Status: Inactive, CRITICAL NLU RESOURCES FAILED TO LOAD."
    else: status_detail = " - Status: Inactive, LIFESPAN DID NOT EXECUTE or failed early."

    return {
        "message": f"{api_version_message}{status_detail}",
        "app_state_lifespan_executed": getattr(request.app.state, 'lifespan_was_executed', "N/A_in_state"),
        "app_state_critical_resources_loaded": getattr(request.app.state, 'critical_resources_loaded', "N/A_in_state"),
        "app_state_symspell_loaded": getattr(request.app.state, 'sym_spell', None) is not None,
        "global_LIFESPAN_EXECUTED_FLAG": LIFESPAN_EXECUTED_FLAG,
        "global_CRITICAL_RESOURCES_LOADED": CRITICAL_RESOURCES_LOADED,
        "global_sym_spell_is_not_none": symspell_loaded_flag,
        "global_sym_spell_word_count": len(symspell_obj.words) if symspell_obj and hasattr(symspell_obj, 'words') else 0,
        "turkish_frequency_dictionary_path_exists": TURKISH_FREQUENCY_DICTIONARY_PATH.exists()
    }