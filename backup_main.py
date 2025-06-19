# main.py
import sys
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from pydantic import BaseModel, Field
import re
import fasttext # type: ignore
import zeyrek # type: ignore
from rapidfuzz import process, fuzz
from pathlib import Path
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import logging
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient
from symspellpy import SymSpell, Verbosity # type: ignore

# Kendi database_service modülünüzü import edin
import database_service

BASE_DIR = Path(__file__).resolve().parent
# Logging seviyesini DEBUG olarak ayarlayarak daha detaylı loglama sağlayabilirsiniz.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv() # .env dosyasındaki değişkenleri yükler

# --- Pydantic Modelleri (Eski kodunuzdan ve güncellemelerle) ---
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    tenant_id: int # Her istekte tenant_id bekleyelim

class NLUSingleAnalysis(BaseModel):
    slm_intent: str
    slm_entities: List[Dict[str, Any]] # Bu alan şu an için kullanılmıyor gibi, ama modelde kalsın.
    confidence_score: float
    message: str

class NLUResponse(BaseModel):
    original_query: str
    processed_query_for_nlu: Optional[str] = None
    session_id: str
    tenant_id: int
    nlu_method: str
    analysis: Optional[NLUSingleAnalysis] = None
    detected_intent: Optional[str] = None
    previous_query_in_session: Optional[str] = None
    resolved_item_details: Optional[Dict[str, Any]] = None # Supabase'den gelen tüm öğe detayı
    resolved_size: Optional[str] = None
    actionable_message: Optional[str] = None # NLU'nun içsel durumu hakkında bilgi
    bot_response: Optional[str] = None
    ask_for_clarification: bool = False
    clarification_options: Optional[List[Dict[str, Any]]] = None # Öğe adı ve ID'si içerebilir: [{"id": 1, "name": "Keten Pantolon"}]


# --- Global Kaynaklar ve Ayarlar (Eski kodunuzdan) ---
MODEL_PATH = (BASE_DIR / "nlu_model.bin").resolve()
TURKISH_FREQUENCY_DICTIONARY_PATH = (BASE_DIR / "turkish_frequency_dictionary.txt").resolve()

# NLU Ayarları ve Kuralları
product_extraction_intents = ["fiyat_sorgulama", "ürün_bilgisi_sorma", "stok_sorgulama", "iade_sorgulama", "ürün_malzeme_sorma"]
# Eski kodunuzdaki tüm kuralları buraya taşıyın
rules = {
    "calisma_saatleri_sorma": re.compile(r"\b((?:çalışma|calisma)\s+saatleri(?:niz)?(?:[\s,.]*nedir\??)?|kaça\s+kadar\s+açık|ne\s+zaman\s+açık|açılış\s+kapanış|mesai|hafta\s*sonu\s+açık|pazar\s+açık\s*mı|hangi\s+saatler|ne\s+zaman\s+kapanıyor|saat\s+kaçta\s+açılıyor|saat\s+kaçta\s+kapanır|açıksınız|calisma\s+saati)\b", re.IGNORECASE),
    "kargo_bilgisi_sorma": re.compile(r"\b(kargo|gönderim|teslimat|kaç\s+günde\s+gelir|kargo\s+ücret|kargo\s+ne\s+kadar|kargo\s+takip|yurtiçi\s+kargo|kargo\s+nekadar|kargonuz\s+kaç\s+günde|teslim\s+süresi|kargo\s+tutar)\b", re.IGNORECASE),
    "fiyat_sorgulama": re.compile(r"\b(fiyat|ücret|kaç\s+para|ne\s+kadar|kaç\s+tl|maliyet|ederi|kaça|nekadar|fyt|fiyt|fyaat|fiyay|kça\s+para|ne\s+kadr|fiyatı\s+ne|fiyatı\s+nedir|ücreti\s+nedir|fiyatını\s+öğren|fiyat\s+bilgisi)(?!.*(?:kargo|teslimat|gönderim|açık|kapanış|saatler\w*|saat|iade|stok|malzeme|özellik|beden|nerede|adres|konum|telefon|mail|ödeme|site|çalışma|calisma|kumaş|içerik)\b)\b", re.IGNORECASE),
    "selamlama": re.compile(r"^\s*(merhaba|selam|iyi\s+günler|günaydın|mrb|slm|sa|selamun\s+aleykum|hey|kolay\s+gelsin|merhba|gunaydn|selamlarr|meraba|s\.a\.|nbr|heyo|selamlar|iyi\s+akşamlar|hayırlı\s+işler)\b", re.IGNORECASE),
    "iade_sorgulama": re.compile(r"\b(iade|geri\s+verme|değişim|değiştir|iade\s+edebilir|iade\s+koşul(?:lar[ıi])?|koşullaeı|para\s+iadesi|değiştirebilir\s+miyim|geri\s+gönderebilir|ürünü\s+geri\s+al|beğenmedim)\b", re.IGNORECASE),
    "stok_sorgulama": re.compile(r"\b(stokta\s+mevcut\s+mu|stokta\s+var\s*m[ıi]|stok\s+durumu|elde\s+var\s+mı|beden[a-zıüöçşğİ.]*\s+var\s*m[ıi]|bednleri\s+var\s*m[ıi]|numarası\s+var\s+mı|modeli\s+var\s+mı|bulunur\s+mu|kaldı\s+mı|bedeni\s+var\s+mı|stok|bedenleri|mevcutmu)(?!.*(?:taksit|ödeme|fiyat|malzeme|kumaş|içerik|saat)\b)\b", re.IGNORECASE),
    "tesekkur": re.compile(r"^\s*(teşekkür\s+ederim|sağ\s+olun|çok\s+teşekkürler|tşk|eyvallah|sağol|teşekkürler|mersi|saol|eyw|tskler|tesekkurler|teşekürler|saolun|varol)\b", re.IGNORECASE),
    "ürün_malzeme_sorma": re.compile(r"\b(malzeme|içerik|kumaş|kumaşı|astar|yapılmış|üretilmiş|neyden\s+yapıl|materyal|kumas\s+ne|içeriğinde\s+ne\s+var|kompozisyonu)(?!.*(?:stok|beden|fiyat|kaç\s+para|ne\s+kadar|bilgi|özellik)\b)\b", re.IGNORECASE),
    "ürün_bilgisi_sorma": re.compile(r"\b(özellikleri|hakkında\s+bilgi|detay|açıklama|nedir\s+bu|ne\s+işe\s+yarar|ürün\s+bilgisi|ürünle\s+ilgili|model\s+hakkında|ürün\s+ne\s+için|anlatır\s+mısın\s+bu\s+ürün|spesifikasyonları)(?!.*(?:malzeme|kumaş|içerik)\b)\b", re.IGNORECASE),
    "lokasyon_sorma": re.compile(r"\b(nerede|adres|konum|yeriniz|mağaza\s+nerede|dükkan\s+nerede|nasıl\s+gel|nerdesiniz|konm|adresiniz\s+neydi|dükkan\s+nerde|hangi\s+semtte|yol\s+tarifi|magazanız)\b", re.IGNORECASE),
    "tel_no_sorma": re.compile(r"\b(telefon|tel\s+no|numara|iletişim\s+no|arayabilir|whatsapp|mail|e-posta|eposta|numaranız|mail\s+adresiniz|irtibat)\b", re.IGNORECASE),
    "odeme_yontemleri_sorma": re.compile(r"\b(nasıl\s+öde|ödeme\s+seçenek|ne\s+kabul|kredi\s+kartı|taksit|kapıda\s+ödeme|havale|eft|ödeme\s+türleri|ödeme\s+yapabilir|taksit\s+imkanı|ödeme\s+şekilleri)(?!.*(?:stok|beden)\b)\b", re.IGNORECASE),
    "websitesi_sorma": re.compile(r"\b(web\s+site|internet\s+site|online\s+mağaza|link|ürünlere\s+nereden\s+bak|sitenizden\s+sipariş|sayfanız|www|site\s+adres|e-ticaret)\b", re.IGNORECASE),
    "musteri_hizmetlerine_baglanma": re.compile(r"\b(müşteri\s+hizmet|yetkili\s+biri|canlı\s+destek|insanla\s+konuş|temsilciye\s+aktar|operatöre\s+bağlan|birine\s+bağla)\b", re.IGNORECASE),
    "siparis_durumu_sorma": re.compile(r"\b(siparişim\s+ne\s+durumda|kargom\s+nerede|siparişimi\s+takip|kargom\s+ne\s+zaman\s+gelir|sipariş\s+no\s+.*\s+ne\s+oldu|ürünüm\s+gelmedi|kargo\s+gelmedi|sipariş\s+durumu)\b", re.IGNORECASE),
    "oneri_isteme": re.compile(r"\b(ne\s+önerirsin|tavsiye\s+eder|en\s+çok\s+satan|benzer\s+ne\s+var|alternatif\s+ne|öneri\s+var\s+mı|ne\s+tavsiye|bir\s+şey\s+öner|hangi\s+ürünü\s+almalı|ne\s+seçmeli)\b", re.IGNORECASE),
    "olumsuz_yanıt": re.compile(r"^\s*(hayır|yok\s+kalsın|gerek\s+yok|istemiyorum|düşünmüyorum|pas|vazgeçtim|kalsın|olmaz|hayr|ilgilenmiyorum|almayayım)\b", re.IGNORECASE),
}
GENERAL_INTENTS_FOR_OVERRIDE = ["selamlama", "tesekkur", "olumsuz_yanıt"]
MIN_WORDS_FOR_SLM_OVERRIDE = 2
SLM_OVERRIDE_CONFIDENCE_THRESHOLD = 0.60 # Bu eşik değerini ayarlayabilirsiniz
FUZZY_MATCH_THRESHOLD = 80 # Bu, veritabanı ILIKE sonrası fallback için kullanılabilir

# Kelime Listeleri (Eski kodunuzdan)
PROTECTED_WORDS_SYMSPELL = {
    "mrb", "slm", "tşk", "eyw", "tmm", "ok", "sa", "kot", "fiyat", "stok", "beden", "ürün", "urun",
    "s", "m", "l", "xl", "xs", "xxl", "xxxl", "small", "medium", "large", "xlarge",
    "bu", "şu", "o", "ne", "mi", "var", "yok", "kaç", "gibi", "göre", "kadar", "için", "ile", "ve", "veya", "ya da", "tl", "try",
    "pantolon", "gömlek", "ceket", "elbise", "etek", "ayakkabı", "model",
    "kırmızı", "mavi", "yeşil", "sarı", "siyah", "beyaz", "pembe", "mor", "turuncu", "gri", "kahverengi", "rengi",
    "fiyatlar", "fiyatları", "bedenler", "bedenleri", "bedeninde", "stokta", "stokları", "stoklar",
    "nedir", "acaba", "miyim", "musunuz", "varmi", "varmı", "mevcutmu", "mevcut", "kaldı",
    "öğrenebilir", "değiştirebilir", "edebilir", "söyler", "alabilir", "olabilir", "yapabilir",
    "ipek", "deri", "keten", "para", "imkanı", "koşulları", "adresiniz", "saatleriniz", "konumunuz",
    "sizde", "sizden", "bana", "sana", "ona", "ol", "hayat"
}
SYMSPELL_BLOCKED_CORRECTIONS = {
    ("ceketin", "çektin"), ("kot", "koy"), ("Kot", "not"), ("fiyatını", "hayatını"), ("fiyatını", "hayat"),
    ("medium", "demek"), ("stokta", "nokta"), ("stok", "sokmak"), ("urun", "uzun"), ("ürünün", "uzunun"),
    ("modelin", "modemin"), ("bednleri", "bebekleri"), ("ceketler", "cesetler"), ("tl", "ol"),
    ("kumas", "kuma"), ("imkanı", "mekanı"), ("ipek", "i pek"), ("ipek", "i̇pek"), ("ipek", "ek"), ("ipek", "i"),
    ("para", "par")
}
KNOWN_TYPOS = {
    "pantoon": "pantolon", "pntolon": "pantolon", "pantalon": "pantolon", "pantln": "pantolon",
    "jekt": "ceket", "jeket": "ceket", "ceketler": "ceket", "kadr": "kadar", "nedr": "nedir",
    "bedn": "beden", "bednleri": "bedenleri", "kırmzı": "kırmızı", "fiyay": "fiyat", "fyt": "fiyat",
    "kça": "kaça", "fiyt": "fiyat", "calısma": "çalışma", "magazanız": "mağazanız", "ürnün": "ürünün",
    "urun": "ürün", "koşullaeı": "koşulları", "merhba": "merhaba", "gunaydn": "günaydın",
    "selamlarr": "selamlar", "meraba": "merhaba", "s.a.": "selamün aleyküm", "nbr": "ne haber",
    "smal": "small", "ktene": "keten", "ketn": "keten", "stokda": "stokta", "varmi": "var mı",
    "mevcutmu": "mevcut mu", "i̇pek": "ipek", "i pek": "ipek", "i ek": "ipek", "iek": "ipek",
    "fiyati": "fiyatı", "taksi": "taksit", "ol": "tl"
}
CORE_WORDS_TO_PRESERVE_LEMMA = {
    "fiyat", "stok", "beden", "iade", "kargo", "adres", "konum", "telefon", "mail", "ödeme", "site",
    "kumaş", "malzeme", "içerik", "özellik", "sipariş", "taksit", "indirim", "kampanya", "ücret", "para",
    "pantolon", "gömlek", "ceket", "elbise", "etek", "ayakkabı", "model", "ürün", "urun",
    "s", "m", "l", "xl", "xs", "xxl", "xxxl", "small", "medium", "large",
    "kırmızı", "mavi", "yeşil", "sarı", "siyah", "beyaz", "pembe", "mor", "turuncu", "gri", "kahverengi", "renk", "rengi",
    "kaç", "ne", "nasıl", "nedir", "kadar", "var", "yok", "bu", "şu", "o", "acaba", "miyim", "misin", "mı", "mi", "mu", "mü",
    "öğrenebilir", "değiştirebilir", "edebilir", "söyler", "mevcut", "kaldı", "alabilir", "olabilir", "yapabilir",
    "fiyatlar", "bedenler", "stoklar", "bedeninde", "stokta", "numara", "numarası", "no", "tl", "try",
    "imkan", "imkanı", "koşul", "koşulları", "ipek", "deri", "keten", "kot"
}

# --- Uygulama Lifespan (Kaynak Yükleme ve Temizleme) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Uygulama başlangıcı: Kaynaklar yükleniyor...")
    app.state.supabase_client = None
    app.state.morphology = None
    app.state.nlu_model = None
    app.state.sym_spell = None
    app.state.critical_resources_loaded = False
    app.state.lifespan_was_executed = True

    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not supabase_url or not supabase_key:
            logger.error("Supabase URL veya Anahtar ortam değişkenlerinde bulunamadı!")
        else:
            app.state.supabase_client = create_client(supabase_url, supabase_key)
            logger.info("Supabase istemcisi başarıyla başlatıldı.")
    except Exception as e:
        logger.error(f"Supabase istemcisi başlatılırken hata: {e}", exc_info=True)

    zeyrek_loaded, fasttext_loaded, symspell_loaded_flag = False, False, False
    try:
        logger.info("Zeyrek MorphAnalyzer yükleniyor...")
        app.state.morphology = zeyrek.MorphAnalyzer()
        logger.info("Zeyrek MorphAnalyzer yüklendi.")
        zeyrek_loaded = True
    except Exception as e: logger.error("Zeyrek yüklenirken KRİTİK HATA: %s", e, exc_info=True)

    if MODEL_PATH.exists():
        try:
            logger.info(f"NLU modeli {MODEL_PATH} adresinden yükleniyor...")
            app.state.nlu_model = fasttext.load_model(str(MODEL_PATH))
            logger.info("NLU modeli yüklendi.")
            fasttext_loaded = True
        except Exception as e: logger.error("NLU modeli yüklenirken KRİTİK HATA: %s", e, exc_info=True)
    else: logger.error(f"NLU model dosyası bulunamadı: {MODEL_PATH}")

    if TURKISH_FREQUENCY_DICTIONARY_PATH.exists():
        try:
            logger.info(f"SymSpell sözlüğü {TURKISH_FREQUENCY_DICTIONARY_PATH} adresinden yükleniyor...")
            temp_sym_spell = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
            if temp_sym_spell.load_dictionary(str(TURKISH_FREQUENCY_DICTIONARY_PATH), term_index=0, count_index=1, separator="\t", encoding="utf-8"):
                app.state.sym_spell = temp_sym_spell
                logger.info(f"SymSpell sözlüğü yüklendi. Kelime sayısı: {len(app.state.sym_spell.words) if app.state.sym_spell else 'N/A'}")
                symspell_loaded_flag = True
            else:
                logger.error(f"SymSpell sözlüğü yüklenemedi: {TURKISH_FREQUENCY_DICTIONARY_PATH}")
        except Exception as e:
            logger.error("SymSpell yüklenirken HATA: %s", e, exc_info=True)
    else:
        logger.warning(f"SymSpell sözlük dosyası bulunamadı: {TURKISH_FREQUENCY_DICTIONARY_PATH}. Yazım denetimi etkilenecek.")

    app.state.critical_resources_loaded = zeyrek_loaded and fasttext_loaded
    logger.info(f"Lifespan özeti: KritikKaynaklarYüklendi={app.state.critical_resources_loaded}, SymSpellYüklendi={symspell_loaded_flag}")

    yield

    logger.info("Uygulama kapanışı: Kaynaklar serbest bırakılıyor...")
    app.state.supabase_client = None # Gerekirse Supabase client'ı kapatma metodu çağrılabilir
    app.state.morphology = None
    app.state.nlu_model = None
    app.state.sym_spell = None
    logger.info("Kaynaklar serbest bırakıldı.")

app = FastAPI(lifespan=lifespan)

# --- NLU Yardımcı Fonksiyonları (Eski kodunuzdan alındı) ---
def _normalize_for_match(text: str) -> str:
    if not text: return ""
    return text.strip().lower().replace("i̇", "i")

def _safe_lemmatize_word(word: str, current_morphology: Optional[zeyrek.MorphAnalyzer]) -> str:
    if not current_morphology:
        return word.lower()

    word_lower = word.lower().strip()
    if not word_lower or word_lower.isdigit():
        return word_lower

    if word_lower == "i̇pek": return "ipek"
    if word_lower == "ipek": return "ipek"
    if word_lower == "iek": return "ipek"
    if word_lower == "tl" : return "tl"
    if word_lower == "para" : return "para"
    if word_lower == "urun" : return "ürün"
    if word_lower == "stok" : return "stok"
    if word_lower == "medium" : return "medium"
    if word_lower == "taksit" : return "taksit"

    if word_lower in CORE_WORDS_TO_PRESERVE_LEMMA:
        return word_lower

    plural_map = {
        "ceketler": "ceket", "gömlekler": "gömlek", "pantolonlar": "pantolon",
        "bedenler": "beden", "fiyatlar": "fiyat", "koşullar": "koşul",
        "özellikler": "özellik", "malzemeler": "malzeme", "renkler": "renk", "stoklar": "stok"
    }
    if word_lower in plural_map:
        return plural_map[word_lower]

    product_bases_for_possessive = [
        "fiyat", "beden", "stok", "model", "ürün", "gömlek", "ceket", "pantolon",
        "kumaş", "malzeme", "içerik", "özellik", "renk", "adres", "konum", "telefon", "imkan", "koşul", "numara"
    ]
    for base in product_bases_for_possessive:
        if word_lower.startswith(base):
            if word_lower == base + "ı" or word_lower == base + "i" or \
               word_lower == base + "u" or word_lower == base + "ü": return base
            if word_lower == base + "sı" or word_lower == base + "si" or \
               word_lower == base + "su" or word_lower == base + "sü": return base
            if len(word_lower) == len(base) + 2 and word_lower.endswith(("ın", "in", "un", "ün")): return base
            if len(word_lower) == len(base) + 3 and word_lower.endswith(("ını", "ini", "unu", "ünü")): return base

    analysis_results = current_morphology.analyze(word_lower)
    if analysis_results and analysis_results[0] and analysis_results[0][0]:
        best_analysis = analysis_results[0][0]
        lemma = best_analysis.lemma.lower()
        pos = best_analysis.pos
        if lemma == "unk" or "Unk" in pos or "Punc" in pos: return word_lower
        if pos not in ["Verb", "Adverb"]:
            if pos in ["Prop", "Abbrv", "Num", "Conj", "Interj", "Postp", "Det", "Ques"]: return word_lower
            if len(lemma) < 3 and word_lower != lemma and not word_lower.isdigit(): return word_lower
            if pos in ["Noun", "Adjective"] and len(lemma) < len(word_lower) - 3 and fuzz.ratio(word_lower, lemma) < 60:
                logger.debug(f"_safe_lemmatize_word: Zeyrek kısa/farklı lemma '{lemma}' -> '{word_lower}' (POS: {pos}). Orijinal kullanılıyor.")
                return word_lower
        logger.debug(f"_safe_lemmatize_word: Zeyrek analizi '{word_lower}': Lemma='{lemma}', POS='{pos}'")
        return lemma
    logger.debug(f"_safe_lemmatize_word: Zeyrek analizi yok '{word_lower}'. Olduğu gibi dönülüyor.")
    return word_lower

def _preprocess_text_for_matching(text_phrase: str, current_morphology: Optional[zeyrek.MorphAnalyzer]) -> str:
    if not text_phrase or not text_phrase.strip(): return ""
    lower_text = text_phrase.lower().strip().replace("i̇", "i")
    lower_text = re.sub(r"\bi\s+(pek|ek)\b", "ipek", lower_text)
    cleaned_text = re.sub(r"[,\.!?\";:]", " ", lower_text)
    cleaned_text = re.sub(r"['`’‘]", "", cleaned_text)
    cleaned_text = re.sub(r"[^\w\s]", " ", cleaned_text) # Harf, rakam ve boşluk dışındakileri boşlukla değiştir
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    words = cleaned_text.split()
    if not current_morphology:
        logger.warning("_preprocess_text_for_matching: Morfoloji analizcisi None. Kelimeler olduğu gibi (küçük harf, temizlenmiş) dönülüyor.")
        return " ".join(words)
    lemmatized_words = [_safe_lemmatize_word(word, current_morphology) for word in words if word]
    final_text = " ".join(lemmatized_words).strip()
    logger.debug(f"_preprocess_text_for_matching: Giriş='{text_phrase}' -> Temiz='{cleaned_text}' -> Lemmatize='{final_text}'")
    return final_text

def correct_spelling(text: str, current_sym_spell: Optional[SymSpell]) -> str:
    if not text or not text.strip(): return text
    corrected_text_intermediate = text
    text_changed_by_known_typos = False
    for typo, correction in KNOWN_TYPOS.items():
        try:
            new_text = re.sub(r'\b' + re.escape(typo) + r'\b', correction, corrected_text_intermediate, flags=re.IGNORECASE)
            if new_text != corrected_text_intermediate:
                corrected_text_intermediate = new_text
                text_changed_by_known_typos = True
        except re.error as e: logger.error(f"KNOWN_TYPOS regex hatası '{typo}': {e}")
    if text_changed_by_known_typos:
        logger.info(f"Bilinen yazım hatası düzeltmesi: Orijinal='{text}' -> Ara='{corrected_text_intermediate}'")

    if not current_sym_spell:
        logger.warning("SymSpell (correct_spelling): SymSpell modeli None. SymSpell tabanlı kontrol atlandı.")
        return corrected_text_intermediate

    words = corrected_text_intermediate.split()
    corrected_words_list = []
    text_changed_by_symspell = False
    size_with_optional_suffix_pattern = re.compile(r"^(\d{2,3}|[SMLX]{1,4})(?:[sıiuüö]{1,2}|si| beden)?$", re.IGNORECASE)
    known_size_abbr = {"S", "M", "L", "XL", "XS", "XXL", "XXXL"}

    for word_val in words:
        original_word_lower = word_val.lower()
        word_to_append = word_val
        if original_word_lower in PROTECTED_WORDS_SYMSPELL:
            corrected_words_list.append(word_val); continue
        
        size_match = size_with_optional_suffix_pattern.match(word_val)
        is_protected_size = False
        if size_match:
            core_part = size_match.group(1).upper()
            if core_part.isdigit():
                try:
                    if 28 <= int(core_part) <= 60: is_protected_size = True
                except ValueError: pass
            elif core_part in known_size_abbr: is_protected_size = True
        if is_protected_size:
            corrected_words_list.append(word_val); continue

        suggestions = current_sym_spell.lookup(word_val, Verbosity.CLOSEST, max_edit_distance=1, include_unknown=True, transfer_casing=True)
        if suggestions:
            best_suggestion = suggestions[0]
            if (original_word_lower, best_suggestion.term.lower()) in SYMSPELL_BLOCKED_CORRECTIONS:
                logger.debug(f"SymSpell: '{word_val}' -> '{best_suggestion.term}' düzeltmesi ENGELLENDİ.")
            elif best_suggestion.term.lower() != original_word_lower and best_suggestion.distance > 0:
                is_safe = True
                # Eski kodunuzdaki özel engelleme ve kontrol mantıkları buraya eklenebilir.
                if len(word_val) > 3 and original_word_lower[0] != best_suggestion.term.lower()[0] and \
                   (original_word_lower not in best_suggestion.term.lower() and best_suggestion.term.lower() not in original_word_lower) and \
                   fuzz.ratio(original_word_lower, best_suggestion.term.lower()) < 75: # Eşik değeri ayarlanabilir
                    logger.debug(f"SymSpell: Riskli düzeltme (ilk harf farklı, düşük benzerlik): '{word_val}' -> '{best_suggestion.term}'. Orijinal korunuyor.")
                    is_safe = False
                if is_safe:
                    word_to_append = best_suggestion.term
                    text_changed_by_symspell = True
            elif best_suggestion.term != word_val : # Sadece case farkı vb.
                word_to_append = best_suggestion.term
                text_changed_by_symspell = True
        corrected_words_list.append(word_to_append)
    
    if text_changed_by_known_typos or text_changed_by_symspell:
        final_corrected_phrase = " ".join(corrected_words_list)
        logger.info(f"Yazım Düzeltme SONUÇ: Orijinal='{text}' -> Düzeltilmiş='{final_corrected_phrase}'")
        return final_corrected_phrase
    return corrected_text_intermediate


def preprocess_query_for_nlu(query: str, current_sym_spell: Optional[SymSpell], current_morphology: Optional[zeyrek.MorphAnalyzer]) -> str:
    spell_checked_query = correct_spelling(query, current_sym_spell)
    lemmatized_query = _preprocess_text_for_matching(spell_checked_query, current_morphology)
    logger.debug(f"preprocess_query_for_nlu: Orijinal='{query}' -> YazımKontrol='{spell_checked_query}' -> Lemmatize='{lemmatized_query}'")
    return lemmatized_query

def extract_simple_entities(original_query_spell_checked: str, processed_query_lemmatized: str,
                            current_morphology: Optional[zeyrek.MorphAnalyzer], # Bu parametre şimdilik kullanılmıyor ama gelecekte gerekebilir
                            intent: Optional[str] = None) -> Dict[str, Any]:
    entities = {"item_name_candidate": None, "size": None, "color": None, "is_generic_product_term": False, "generic_term_options": []}
    logger.debug(f"--- Varlık Çıkarımı Başladı ---")
    logger.debug(f"Giriş (yazım kontrol edilmiş): '{original_query_spell_checked}'")
    logger.debug(f"Giriş (lematize edilmiş): '{processed_query_lemmatized}'")
    logger.debug(f"Giriş (niyet): '{intent}'")

    # Beden Çıkarımı (Eski kodunuzdaki gibi)
    query_for_size_cleaned = re.sub(r"\bvar\s+m[ıi]?\b", " var_soru ", original_query_spell_checked, flags=re.IGNORECASE)
    size_pattern = r"\b(\d{2,3}|(?:X?S|M|L|X{1,3}L)|small|medium|large|xsmall|xlarge|xxlarge|ekstra\s+large|x\s+large|sml|smal)\b"
    size_match = re.search(size_pattern, query_for_size_cleaned, re.IGNORECASE)
    if size_match:
        extracted_size_token = size_match.group(1).lower()
        core_size_token = extracted_size_token.upper()
        size_normalization_map = {
            "SMALL": "S", "MEDIUM": "M", "LARGE": "L", "XSMALL": "XS", "XLARGE": "XL",
            "XXLARGE": "XXL", "XXXL": "XXXL", "EKSTRA LARGE": "XL", "X LARGE": "XL", "SML": "S", "SMAL": "S"
        }
        if core_size_token.isdigit():
            try:
                if 28 <= int(core_size_token) <= 60: entities["size"] = core_size_token
            except ValueError: pass
        elif core_size_token in size_normalization_map: entities["size"] = size_normalization_map[core_size_token]
        elif core_size_token in ["S", "M", "L", "XL", "XXL", "XS", "XXXL"]: entities["size"] = core_size_token
        if entities.get("size"): logger.info(f"Beden çıkarıldı: '{entities['size']}' token'dan '{extracted_size_token}'")

    # Öğe Adı Adayı Çıkarımı
    # Bu kısım, veritabanı aramasında kullanılacak bir "aday" metin çıkarmalıdır.
    # Eski kodunuzdaki gibi, ürün adını veya genel terimi (pantolon, gömlek vb.) belirlemeye çalışır.
    # Supabase'de arama yapacağımız için, bu fonksiyonun görevi, arama için makul bir string üretmektir.
    # Şimdilik, eski kodunuzdaki mantığa benzer bir şekilde, sorgudan anahtar kelimeleri temizleyerek bir aday çıkaralım.
    # Bu, `database_service.get_items_by_name_fuzzy` fonksiyonuna gönderilecek.
    
    # Basit bir "öğe adı adayı" bulma:
    # "fiyat", "stok" gibi kelimelerden önceki veya sorgunun başındaki kelime grupları.
    # Bu, çok kural tabanlı ve kırılgandır. Daha iyi bir model için Named Entity Recognition (NER)
    # veya veritabanı ile fuzzy eşleştirme daha robust olur.
    
    # Eski kodunuzdaki `product_phrase_for_fuzz_cleaned` mantığını buraya taşıyalım.
    # Bu, `original_query_spell_checked` üzerinden çalışır.
    product_phrase_for_fuzz = original_query_spell_checked
    temp_words = product_phrase_for_fuzz.split()
    keywords_to_strip_for_fuzz = {
        "fiyat", "fiyatı", "fiyatını", "nedir", "ne kadar", "var mı", "beden", "stok",
        "özellikleri", "kumaşı", "malzemesi", "içeriği", "içeriğinde",
        "acaba", "mı", "mi", "mu", "mü", "kaç", "para", "sizde", "lütfen", "ya",
        "kırmızı", "mavi", "yeşil", "sarı", "siyah", "beyaz", "rengi" # Renkler ayrı bir entity olabilir
    }
    # Baştan ve sondan keywordleri temizle
    while temp_words and temp_words[-1].lower().strip("?.,!") in keywords_to_strip_for_fuzz: temp_words.pop()
    while temp_words and temp_words[0].lower().strip("?.,!") in ["ya", "acaba", "peki", "hadi", "kolay", "gelsin", "merhaba", "selam"]: temp_words.pop(0)
    
    product_phrase_for_fuzz_cleaned = " ".join(temp_words).strip().rstrip(',.?!')

    if product_phrase_for_fuzz_cleaned:
        entities["item_name_candidate"] = product_phrase_for_fuzz_cleaned
        logger.info(f"Potansiyel öğe adı adayı (fuzzy için): '{entities['item_name_candidate']}'")
    else:
        # Fallback: Eğer temizleme sonrası bir şey kalmazsa, orijinal sorgunun bir kısmını alabiliriz.
        # Veya `processed_query_lemmatized` kullanılabilir.
        # Şimdilik, `processed_query_lemmatized`'i aday olarak alalım.
        entities["item_name_candidate"] = processed_query_lemmatized
        logger.warning(f"Fuzzy için aday çıkarılamadı, lemmatize edilmiş sorgu aday olarak kullanılıyor: '{entities['item_name_candidate']}'")
    
    # Generic term detection (pantolon, gömlek vb.)
    # Bu, `item_name_candidate` Supabase'de arandıktan sonra, eğer tam bir ürün bulunamazsa
    # ama `item_name_candidate` bir kategoriye (pantolon, gömlek) benziyorsa devreye girebilir.
    # Şimdilik bu kısmı `process_query` içinde, Supabase yanıtına göre ele alalım.

    logger.info(f"--- Varlık Çıkarımı Tamamlandı --- Varlıklar: {entities}")
    return entities


def call_slm_model(processed_query_for_slm: str, current_nlu_model: Optional[fasttext.FastText._FastText]) -> NLUSingleAnalysis:
    if not current_nlu_model:
        logger.error("SLM çağrısı başarısız: NLU modeli None.")
        return NLUSingleAnalysis(slm_intent="hata_slm_model_yok", slm_entities=[], confidence_score=0.0, message="SLM (fastText) modeli None.")
    if not processed_query_for_slm or not processed_query_for_slm.strip() or len(processed_query_for_slm.strip()) < 2 :
        logger.warning(f"SLM çağrısı atlandı: SLM için işlenmiş sorgu çok kısa veya boş: '{processed_query_for_slm}'")
        return NLUSingleAnalysis(slm_intent="tahmin_yok_slm_kisa_sorgu", slm_entities=[], confidence_score=0.0, message="Sorgu SLM için çok kısa.")

    cleaned_query_for_slm = processed_query_for_slm.replace("\n", " ")
    logger.debug(f"SLM modeline gönderilen sorgu: '{cleaned_query_for_slm}'")
    predictions = current_nlu_model.predict(cleaned_query_for_slm, k=1)
    intent_name = "tahmin_yok_slm_ile"; confidence = 0.0
    if predictions and predictions[0] and predictions[1] and predictions[0][0] and predictions[1][0]:
        predicted_label_full = predictions[0][0]; confidence = predictions[1][0]
        intent_name = predicted_label_full.replace("__label__", "")
    logger.info(f"SLM MODEL ÇAĞRISI: Sorgu='{processed_query_for_slm}' -> Niyet: {intent_name}, Güven: {confidence:.4f}")
    return NLUSingleAnalysis(slm_intent=intent_name, slm_entities=[], confidence_score=float(f"{confidence:.4f}"), message="fastText SLM modelinden yanıt.")


# --- Ana Sorgu İşleme Endpoint'i ---
@app.post("/process_query/", response_model=NLUResponse)
async def process_query(payload: QueryRequest, request: FastAPIRequest):
    start_time = datetime.now(timezone.utc)
    original_user_query = payload.query
    tenant_id = payload.tenant_id
    effective_session_id = payload.session_id or str(uuid.uuid4())

    # Kaynakları app.state'den al
    supabase: Optional[SupabaseClient] = getattr(request.app.state, 'supabase_client', None)
    current_sym_spell: Optional[SymSpell] = getattr(request.app.state, 'sym_spell', None)
    current_morphology: Optional[zeyrek.MorphAnalyzer] = getattr(request.app.state, 'morphology', None)
    current_nlu_model: Optional[fasttext.FastText._FastText] = getattr(request.app.state, 'nlu_model', None)

    if not getattr(request.app.state, 'lifespan_was_executed', False) or \
       not getattr(request.app.state, 'critical_resources_loaded', False) or \
       not supabase:
        logger.error("Kritik sistem kaynakları yüklenemedi veya Supabase istemcisi yok.")
        # Bu durumda loglama için Supabase kullanılamaz.
        raise HTTPException(status_code=503, detail="Sistem hatası: Temel NLU kaynakları veya veritabanı bağlantısı yüklenemedi.")

    if not original_user_query or not original_user_query.strip():
        raise HTTPException(status_code=400, detail="Sorgu boş olamaz.")

    logger.info(f"--- Yeni Sorgu Alındı: '{original_user_query}', Tenant ID: {tenant_id}, Session ID: {effective_session_id} ---")

    # 1. Kiracı Ayarlarını Çek
    tenant_settings = await database_service.get_tenant_settings(supabase, tenant_id)
    if not tenant_settings:
        logger.error(f"Tenant {tenant_id} için ayarlar alınamadı.")
        # Loglama
        log_entry_minimal = { "tenant_id": tenant_id, "session_id": effective_session_id, "user_query_original": original_user_query, "bot_response": "İşletme bilgileri alınamadı.", "error_message": "Tenant ayarları bulunamadı.", "created_at": datetime.now(timezone.utc).isoformat()}
        if supabase: await database_service.log_conversation_turn(supabase, log_entry_minimal) # Supabase varsa logla
        raise HTTPException(status_code=404, detail=f"İşletme (ID: {tenant_id}) bulunamadı veya ayarları eksik.")

    business_type = tenant_settings.get("business_type", "unknown") # Varsayılan tür
    business_name = tenant_settings.get("business_name", "İşletmemiz")
    settings_json = tenant_settings.get("settings_json", {})
    default_responses = settings_json.get("default_responses", {})
    currency = settings_json.get("currency", "TL")
    business_info_from_settings = settings_json.get("business_info", {}) # Eski simulated_business_data['business_info'] yerine

    item_not_found_msg_template = default_responses.get("item_not_found", "Üzgünüm, '{item_name}' hakkında bir bilgim bulunmuyor. Size {business_name} olarak başka nasıl yardımcı olabilirim?")
    general_error_msg = default_responses.get("general_error", "Üzgünüm, bir sorun oluştu. Lütfen daha sonra tekrar deneyin.")
    clarification_needed_msg = default_responses.get("clarification_needed", "Hangi {item_type} hakkında bilgi almak istiyorsunuz?")
    out_of_scope_msg = default_responses.get("out_of_scope", "Üzgünüm, bu konuda yardımcı olamıyorum. Ürünlerimiz veya hizmetlerimizle ilgili soru sorabilirsiniz.")
    greeting_msg = default_responses.get("greeting", f"Merhaba! {business_name} olarak size nasıl yardımcı olabilirim?")
    thanks_response_msg = default_responses.get("thanks_response", "Rica ederim! Başka bir konuda yardımcı olabilir miyim?")
    fallback_msg = default_responses.get("fallback", "Ne demek istediğinizi tam anlayamadım. Farklı kelimelerle tekrar sorabilir misiniz?")


    # 2. Oturum Verilerini Çek (Supabase'den)
    session_data = await database_service.get_session_data(supabase, effective_session_id, tenant_id)
    if session_data is None: # Veri çekilemediyse (hata olduysa)
        logger.error(f"Oturum ({effective_session_id}) verisi çekilirken hata oluştu. Yeni oturum gibi devam edilecek.")
        session_data = {"history": [], "last_mentioned_item_id": None, "intent_awaiting_clarification": None, "clarification_options_offered": None}
    elif not session_data: # Boş dict döndüyse (yeni oturum)
        logger.info(f"Oturum ({effective_session_id}) için veri bulunamadı, yeni oturum başlatılıyor.")
        session_data = {"history": [], "last_mentioned_item_id": None, "intent_awaiting_clarification": None, "clarification_options_offered": None}
    
    previous_query_text = session_data.get("history", [])[-1].get("query") if session_data.get("history") else None


    # 3. NLU İşlemleri
    spell_checked_query = correct_spelling(original_user_query, current_sym_spell)
    processed_query_for_nlu = preprocess_query_for_nlu(original_user_query, current_sym_spell, current_morphology)

    final_intent = None
    nlu_method = "unknown"
    slm_analysis_result: Optional[NLUSingleAnalysis] = None
    resolved_item_from_db: Optional[Dict[str, Any]] = None
    ask_for_clarification_flag = False
    clarification_options_list: Optional[List[Dict[str, Any]]] = None
    bot_response_text = general_error_msg # Varsayılan

    # 3a. Bağlam Yönetimi (Netleştirme) - Eski kodunuzdaki mantıkla benzer
    intent_awaiting_clarification = session_data.get("intent_awaiting_clarification")
    clarification_options_offered_ctx = session_data.get("clarification_options_offered") # [{id:1, name:"A"}, {id:2, name:"B"}] formatında olmalı

    user_reply_normalized_for_clarif = _normalize_for_match(original_user_query) # Kullanıcının yanıtı
    matched_clarification_option_id: Optional[Any] = None

    if intent_awaiting_clarification and isinstance(clarification_options_offered_ctx, list):
        for option_detail in clarification_options_offered_ctx:
            if isinstance(option_detail, dict) and "name" in option_detail and "id" in option_detail:
                normalized_option_name = _normalize_for_match(str(option_detail["name"])) # DB'den gelen isim
                # Basit eşleşme: Kullanıcı tam adı yazdıysa
                if normalized_option_name == user_reply_normalized_for_clarif:
                    matched_clarification_option_id = option_detail["id"]
                    logger.info(f"Netleştirme yanıtı EŞLEŞTİ (isimle): Seçenek Adı='{option_detail['name']}', ID='{matched_clarification_option_id}'")
                    break
                # TODO: Kullanıcı "birincisi", "ikincisi" veya sadece numara yazdıysa da ele alınabilir.
        
        if matched_clarification_option_id:
            final_intent = intent_awaiting_clarification
            nlu_method = f"contextual_clarification_for_{final_intent}"
            resolved_item_from_db = await database_service.get_item_by_id(supabase, tenant_id, business_type, matched_clarification_option_id)
            if resolved_item_from_db:
                logger.info(f"Netleştirme ile öğe çözüldü: {resolved_item_from_db.get('name')}")
                session_data["last_mentioned_item_id"] = resolved_item_from_db.get("id")
            else:
                logger.warning(f"Netleştirme ile ID ({matched_clarification_option_id}) bulundu ama öğe DB'den çekilemedi.")
            # Netleştirme durumunu temizle
            session_data["intent_awaiting_clarification"] = None
            session_data["clarification_options_offered"] = None


    # 3b. Normal NLU Akışı (Eğer netleştirme ile çözülmediyse veya niyet hala yoksa)
    if not final_intent: # final_intent hala None ise (netleştirme ile belirlenmediyse)
        query_for_regex = spell_checked_query.lower() # Yazım kontrolü yapılmış sorgu
        detected_intent_via_regex = None
        intent_priority_order = [ # Eski kodunuzdaki öncelik sırası
            "selamlama", "tesekkur", "olumsuz_yanıt", "musteri_hizmetlerine_baglanma", "siparis_durumu_sorma",
            "calisma_saatleri_sorma", "kargo_bilgisi_sorma", "lokasyon_sorma", "tel_no_sorma",
            "odeme_yontemleri_sorma", "websitesi_sorma", "iade_sorgulama",
            "fiyat_sorgulama", "stok_sorgulama", "ürün_malzeme_sorma", "ürün_bilgisi_sorma", "oneri_isteme"
        ]
        for intent_key in intent_priority_order:
            pattern = rules.get(intent_key)
            if not pattern: continue
            # Eski kodunuzdaki gibi match veya search kullanın
            match_condition = pattern.match if intent_key in GENERAL_INTENTS_FOR_OVERRIDE else pattern.search
            if match_condition(query_for_regex):
                # Eski kodunuzdaki çakışma kontrolleri (fiyat vs çalışma saati vb.)
                if intent_key == "fiyat_sorgulama" and rules.get("calisma_saatleri_sorma", re.compile("^(?!.*)$")).search(query_for_regex): continue
                if intent_key == "stok_sorgulama" and rules.get("odeme_yontemleri_sorma", re.compile("^(?!.*)$")).search(query_for_regex): continue
                # ... diğer çakışma kontrolleri ...
                detected_intent_via_regex = intent_key
                logger.info(f"Niyet Regex ile bulundu: {detected_intent_via_regex}")
                break
        
        # SLM Modeli ile niyet tespiti (Eski kodunuzdaki SLM override mantığı)
        if detected_intent_via_regex in GENERAL_INTENTS_FOR_OVERRIDE and len(original_user_query.split()) >= MIN_WORDS_FOR_SLM_OVERRIDE:
            slm_analysis_result = call_slm_model(processed_query_for_nlu, current_nlu_model)
            if slm_analysis_result and slm_analysis_result.slm_intent not in GENERAL_INTENTS_FOR_OVERRIDE and \
               slm_analysis_result.slm_intent not in ["kapsam_disi", "tahmin_yok_slm_ile", "hata_slm_model_yok", "tahmin_yok_slm_kisa_sorgu"] and \
               slm_analysis_result.confidence_score >= SLM_OVERRIDE_CONFIDENCE_THRESHOLD:
                final_intent = slm_analysis_result.slm_intent
                nlu_method = f"slm_override_of_{detected_intent_via_regex}"
            else:
                final_intent = detected_intent_via_regex
                nlu_method = f"regex_kept_{detected_intent_via_regex}"
        elif detected_intent_via_regex:
            final_intent = detected_intent_via_regex
            nlu_method = f"regex_specific_{final_intent}"
        else: # Regex bir şey bulamadıysa direkt SLM
            slm_analysis_result = call_slm_model(processed_query_for_nlu, current_nlu_model)
            if slm_analysis_result:
                final_intent = slm_analysis_result.slm_intent
                nlu_method = f"slm_direct_{final_intent}"
                # SLM sonucu güvenilir değilse kapsam dışı yap (eski kodunuzdaki mantık)
                if final_intent in ["tahmin_yok_slm_ile", "hata_slm_model_yok", "tahmin_yok_slm_kisa_sorgu"] or \
                   (final_intent == "kapsam_disi" and slm_analysis_result.confidence_score < 0.5) or \
                   (slm_analysis_result.confidence_score < 0.35 and final_intent not in GENERAL_INTENTS_FOR_OVERRIDE and final_intent not in product_extraction_intents):
                    logger.warning(f"SLM sonucu '{final_intent}' (Güven: {slm_analysis_result.confidence_score:.2f}) güvenilir değil. 'kapsam_disi' olarak ayarlanıyor.")
                    final_intent = "kapsam_disi"
                    nlu_method += "_fallback_to_kapsam_disi"
            else: # SLM de bir şey döndürmediyse
                 final_intent = "kapsam_disi"
                 nlu_method = "fallback_slm_failed_to_kapsam_disi"


        if not final_intent: # Hala niyet yoksa
            final_intent = "kapsam_disi"
            nlu_method = "ultimate_fallback_to_kapsam_disi"
        
        logger.info(f"Nihai Niyet: {final_intent}, Yöntem: {nlu_method}")

        # 3c. Varlık Çıkarımı (Eski kodunuzdaki extract_simple_entities çağrısı)
        current_entities = extract_simple_entities(spell_checked_query, processed_query_for_nlu, current_morphology, final_intent)
        item_name_candidate_from_entities = current_entities.get("item_name_candidate")
        resolved_size_entity = current_entities.get("size")
        # is_generic_product_term_from_entities = current_entities.get("is_generic_product_term", False) # Bu bilgi DB aramasından sonra daha anlamlı olacak

        # 3d. Veritabanından Öğe Arama (Eğer niyet ürünle ilgiliyse ve netleştirme ile çözülmediyse)
        if final_intent in product_extraction_intents and not resolved_item_from_db: # Henüz bir ürün çözülmediyse
            if item_name_candidate_from_entities:
                items_found_in_db = await database_service.get_items_by_name_fuzzy(supabase, tenant_id, business_type, item_name_candidate_from_entities, limit=5)
                
                if items_found_in_db:
                    if len(items_found_in_db) == 1:
                        resolved_item_from_db = items_found_in_db[0]
                        logger.info(f"Öğe veritabanında bulundu (tek eşleşme): {resolved_item_from_db.get('name')}")
                        session_data["last_mentioned_item_id"] = resolved_item_from_db.get("id")
                    else: # Birden fazla öğe bulundu, netleştirme sor
                        ask_for_clarification_flag = True
                        # Seçenekler ID ve isim içermeli
                        clarification_options_list = [{"id": item.get("id"), "name": item.get("name")} for item in items_found_in_db if item.get("id") and item.get("name")]
                        
                        if clarification_options_list: # Gerçekten seçenek varsa
                            item_names_for_clarification = ", ".join([f"'{opt['name']}'" for opt in clarification_options_list])
                            bot_response_text = f"'{item_name_candidate_from_entities}' ile ilgili birkaç seçenek buldum: {item_names_for_clarification}. Hangisini sormuştunuz?"
                            session_data["intent_awaiting_clarification"] = final_intent
                            session_data["clarification_options_offered"] = clarification_options_list
                            logger.info(f"Birden fazla öğe bulundu, netleştirme gerekiyor: {item_names_for_clarification}")
                        else: # Seçenek oluşturulamadıysa (örn. isimleri yoktu)
                            logger.warning("Birden fazla öğe bulundu ama netleştirme seçenekleri oluşturulamadı.")
                            # Bu durumda "bulunamadı" gibi davranılabilir.
                            bot_response_text = item_not_found_msg_template.format(item_name=item_name_candidate_from_entities, business_name=business_name)

                else: # DB'de öğe bulunamadı
                    logger.info(f"'{item_name_candidate_from_entities}' adayı için veritabanında öğe bulunamadı.")
                    # bot_response_text item_not_found ile ayarlanacak (yanıt oluşturma kısmında)
            
            elif session_data.get("last_mentioned_item_id"): # Öğe adayı yok ama oturumda önceki öğe var
                last_item_id = session_data.get("last_mentioned_item_id")
                if last_item_id:
                    resolved_item_from_db = await database_service.get_item_by_id(supabase, tenant_id, business_type, last_item_id)
                    if resolved_item_from_db:
                        logger.info(f"Oturumdan önceki öğe kullanılıyor: {resolved_item_from_db.get('name')}")
                    else: # Önceki ID artık geçerli değilse
                        session_data["last_mentioned_item_id"] = None # Temizle
                        logger.warning(f"Oturumdaki önceki öğe ID ({last_item_id}) artık DB'de bulunamadı.")


    # --- 4. Yanıt Oluşturma ---
    actionable_message = f"Intent: {final_intent or 'belirlenemedi'}."
    if resolved_item_from_db: actionable_message += f" ItemID: {resolved_item_from_db.get('id')}, ItemName: {resolved_item_from_db.get('name')}."
    elif 'item_name_candidate_from_entities' in locals() and item_name_candidate_from_entities: actionable_message += f" ItemCandidate: {item_name_candidate_from_entities}."
    if 'resolved_size_entity' in locals() and resolved_size_entity: actionable_message += f" Size: {resolved_size_entity}."

    if not ask_for_clarification_flag: # Eğer zaten netleştirme sormuyorsak normal yanıt oluştur
        if final_intent == "fiyat_sorgulama":
            if resolved_item_from_db:
                price = resolved_item_from_db.get("price") # Bu, doğrudan price kolonu veya attributes_json.price olabilir
                item_name = resolved_item_from_db.get("name", "Bu ürün")
                attributes = resolved_item_from_db.get("attributes_json", {})
                if price is None: price = attributes.get("price") # attributes_json'dan da kontrol et

                if price is not None:
                    bot_response_text = f"'{item_name}' için fiyat: {price} {currency}."
                else:
                    bot_response_text = f"'{item_name}' için fiyat bilgisi bulunamadı."
            elif item_name_candidate_from_entities:
                 bot_response_text = item_not_found_msg_template.format(item_name=item_name_candidate_from_entities, business_name=business_name)
            else:
                bot_response_text = clarification_needed_msg.format(item_type="ürünün") + " Fiyatını öğrenmek istediğiniz ürünün adını söyler misiniz?"
        
        elif final_intent == "stok_sorgulama":
            if resolved_item_from_db:
                item_name = resolved_item_from_db.get("name", "Bu ürün")
                attributes = resolved_item_from_db.get("attributes_json", {})
                # Stok bilgisi 'attributes_json' içinde 'available_sizes_info', 'stock_quantity' gibi alanlarda olabilir.
                # Eski kodunuzdaki 'available_sizes_info' mantığını kullanalım (Supabase'de attributes_json içinde olduğunu varsayarak)
                stock_info = attributes.get("available_sizes_info", "Stok durumu ve bedenler için lütfen ürün detaylarına bakınız.")
                link = attributes.get("link", "") # Link de attributes_json'da olabilir
                
                response_parts = []
                if resolved_size_entity:
                    response_parts.append(f"'{item_name}' ({resolved_size_entity} beden) için stok durumu: {stock_info}.")
                else:
                    response_parts.append(f"'{item_name}' için stok durumu: {stock_info}.")
                if link: response_parts.append(f"Detaylar: {link}")
                bot_response_text = " ".join(response_parts)
            elif item_name_candidate_from_entities:
                 bot_response_text = item_not_found_msg_template.format(item_name=item_name_candidate_from_entities, business_name=business_name)
            else:
                bot_response_text = clarification_needed_msg.format(item_type="ürünün") + " Stok durumunu merak ettiğiniz ürünün adını ve varsa bedenini belirtir misiniz?"

        elif final_intent == "ürün_bilgisi_sorma":
            if resolved_item_from_db:
                item_name = resolved_item_from_db.get("name", "Bu ürün")
                attributes = resolved_item_from_db.get("attributes_json", {})
                description = attributes.get("description", "Bu ürün hakkında detaylı açıklama bulunmamaktadır.")
                link = attributes.get("link", "")
                bot_response_text = f"'{item_name}': {description}"
                if link: bot_response_text += f" Detaylar: {link}"
            elif item_name_candidate_from_entities:
                 bot_response_text = item_not_found_msg_template.format(item_name=item_name_candidate_from_entities, business_name=business_name)
            else:
                bot_response_text = clarification_needed_msg.format(item_type="ürün") + " Hangi ürün hakkında bilgi almak istiyorsunuz?"

        elif final_intent == "ürün_malzeme_sorma":
            if resolved_item_from_db:
                item_name = resolved_item_from_db.get("name", "Bu ürün")
                attributes = resolved_item_from_db.get("attributes_json", {})
                material = attributes.get("material_composition", "Malzeme bilgisi belirtilmemiş.")
                link = attributes.get("link", "")
                bot_response_text = f"'{item_name}' ürününün malzeme içeriği: {material}."
                if link: bot_response_text += f" Detaylar: {link}"
            elif item_name_candidate_from_entities:
                 bot_response_text = item_not_found_msg_template.format(item_name=item_name_candidate_from_entities, business_name=business_name)
            else:
                bot_response_text = clarification_needed_msg.format(item_type="ürünün") + " Hangi ürünün malzeme bilgisini öğrenmek istiyorsunuz?"
        
        elif final_intent == "iade_sorgulama":
            # İade politikası tenant_settings.settings_json.business_info içinde olabilir
            return_policy = business_info_from_settings.get("return_policy", "İade politikamız hakkında detaylı bilgi için lütfen web sitemizi ziyaret edin veya müşteri hizmetlerimizle iletişime geçin.")
            if resolved_item_from_db:
                item_name = resolved_item_from_db.get("name", "Bu ürün")
                bot_response_text = f"'{item_name}' ile ilgili iade politikamız: {return_policy}"
            else:
                bot_response_text = f"Genel iade politikamız: {return_policy}"

        # Genel Niyetler (Eski kodunuzdaki gibi, business_info_from_settings kullanarak)
        elif final_intent == "selamlama": bot_response_text = greeting_msg
        elif final_intent == "tesekkur": bot_response_text = thanks_response_msg
        elif final_intent == "olumsuz_yanıt": bot_response_text = default_responses.get("negative_response_ack", "Anladım. Başka bir konuda yardımcı olabilir miyim?") # settings_json'dan
        elif final_intent == "lokasyon_sorma":
            address = business_info_from_settings.get("address", "Adres bilgimiz şu an mevcut değil.")
            maps_link = business_info_from_settings.get("maps_link", "")
            bot_response_text = f"Adresimiz: {address}."
            if maps_link: bot_response_text += f" Harita: {maps_link}"
        elif final_intent == "tel_no_sorma":
            phone = business_info_from_settings.get("phone", "")
            whatsapp = business_info_from_settings.get("whatsapp_number", "")
            email = business_info_from_settings.get("email", "")
            parts = []
            if phone: parts.append(f"Telefon: {phone}")
            if whatsapp: parts.append(f"WhatsApp: {whatsapp}")
            if email: parts.append(f"E-posta: {email}")
            if parts: bot_response_text = ". ".join(parts) + "."
            else: bot_response_text = "İletişim bilgilerimiz şu an mevcut değil."
        elif final_intent == "calisma_saatleri_sorma":
            hours = business_info_from_settings.get("opening_hours", "Çalışma saatlerimiz hakkında bilgi için lütfen bizimle iletişime geçin.")
            bot_response_text = f"Çalışma saatlerimiz: {hours}."
        elif final_intent == "odeme_yontemleri_sorma":
            payment = business_info_from_settings.get("payment_options", "Ödeme yöntemlerimiz hakkında bilgi için lütfen bizimle iletişime geçin.")
            bot_response_text = f"Kabul ettiğimiz ödeme yöntemleri: {payment}."
        elif final_intent == "kargo_bilgisi_sorma":
            shipping = business_info_from_settings.get("shipping_info", "Kargo bilgilerimiz için lütfen bizimle iletişime geçin.")
            bot_response_text = shipping
        elif final_intent == "websitesi_sorma":
            website = business_info_from_settings.get("website", "")
            if website: bot_response_text = f"Web sitemizden tüm ürünlerimize ulaşabilirsiniz: {website}"
            else: bot_response_text = "Web sitemizin adresi şu an mevcut değil."
        elif final_intent == "musteri_hizmetlerine_baglanma":
            phone_mh = business_info_from_settings.get("phone", "müşteri hizmetleri numaramızdan")
            bot_response_text = f"Müşteri hizmetlerimize {phone_mh} ulaşabilirsiniz."
        elif final_intent == "siparis_durumu_sorma":
            bot_response_text = default_responses.get("order_status_info", "Sipariş durumunuzu web sitemizdeki hesabınızdan veya müşteri hizmetlerimizden öğrenebilirsiniz.")
        elif final_intent == "oneri_isteme":
            bot_response_text = default_responses.get("recommendation_prompt", "Tabii, ne tür bir ürün arıyorsunuz? Size yardımcı olmaktan mutluluk duyarım.")
        
        # Kapsam Dışı ve Fallback
        elif final_intent == "kapsam_disi":
            bot_response_text = out_of_scope_msg
        elif final_intent in ["tahmin_yok_slm_ile", "hata_slm_model_yok", "tahmin_yok_slm_kisa_sorgu"]:
            bot_response_text = fallback_msg
        else: # Kapsanmayan bir durum veya genel hata (eğer yukarıda atanmadıysa)
            if not bot_response_text or bot_response_text == general_error_msg :
                 bot_response_text = fallback_msg


    # --- 5. Konuşma Geçmişini ve Oturum Verilerini Güncelle/Kaydet ---
    current_turn_log_for_session = { # Oturum geçmişi için daha kısa bir log
        "query": original_user_query,
        "bot_response": bot_response_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intent": final_intent, # Basit niyet bilgisi
        "item_id_resolved": resolved_item_from_db.get("id") if resolved_item_from_db else None
    }
    session_history = session_data.get("history", [])
    session_history.append(current_turn_log_for_session)
    session_data["history"] = session_history[-5:] # Son 5 etkileşimi tut (isteğe bağlı)

    # `last_mentioned_item_id` zaten yukarıda güncellendi.
    # `intent_awaiting_clarification` ve `clarification_options_offered` da yukarıda güncellendi/temizlendi.
    await database_service.save_session_data(supabase, effective_session_id, tenant_id, session_data)

    # --- 6. Detaylı Konuşma Adımını Logla (Ayrı bir tabloya) ---
    log_entry_for_db = {
        "tenant_id": tenant_id,
        "session_id": effective_session_id,
        "user_query_original": original_user_query,
        "user_query_spell_checked": spell_checked_query,
        "user_query_lemmatized": processed_query_for_nlu,
        "detected_intent": final_intent,
        "nlu_method": nlu_method,
        "slm_intent": slm_analysis_result.slm_intent if slm_analysis_result else None,
        "slm_confidence": slm_analysis_result.confidence_score if slm_analysis_result else None,
        "entities_extracted": { # current_entities'den gelenler
            "item_name_candidate": current_entities.get("item_name_candidate") if 'current_entities' in locals() else None,
            "size": current_entities.get("size") if 'current_entities' in locals() else None,
            "color": current_entities.get("color") if 'current_entities' in locals() else None
        },
        "resolved_item_id": resolved_item_from_db.get("id") if resolved_item_from_db else None,
        "resolved_item_name": resolved_item_from_db.get("name") if resolved_item_from_db else None, # DB'den gelen asıl isim
        "bot_response": bot_response_text,
        "ask_for_clarification": ask_for_clarification_flag,
        "clarification_options_offered": clarification_options_list, # Sadece isimleri veya ID'leri loglamak daha iyi olabilir
        "response_time_ms": (datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
        "created_at": start_time.isoformat()
    }
    await database_service.log_conversation_turn(supabase, log_entry_for_db)

    logger.info(f"Yanıt Gönderiliyor: '{bot_response_text}'")
    return NLUResponse(
        original_query=original_user_query,
        processed_query_for_nlu=processed_query_for_nlu,
        session_id=effective_session_id,
        tenant_id=tenant_id,
        nlu_method=nlu_method,
        analysis=slm_analysis_result,
        detected_intent=final_intent,
        previous_query_in_session=previous_query_text,
        resolved_item_details=resolved_item_from_db,
        resolved_size=resolved_size_entity if 'resolved_size_entity' in locals() else None,
        actionable_message=actionable_message,
        bot_response=bot_response_text,
        ask_for_clarification=ask_for_clarification_flag,
        clarification_options=clarification_options_list
    )

@app.get("/")
async def read_root(request: FastAPIRequest):
    # Eski kodunuzdaki root endpoint'i
    api_version_message = "Chatbot NLU API (v1.0 - Supabase Entegre & Birleştirilmiş)" # Versiyonu güncelleyin
    lifespan_run = getattr(request.app.state, 'lifespan_was_executed', False)
    critical_loaded = getattr(request.app.state, 'critical_resources_loaded', False)
    symspell_from_state = getattr(request.app.state, 'sym_spell', None)
    symspell_is_loaded_via_state = symspell_from_state is not None
    symspell_word_count_via_state = len(symspell_from_state.words) if symspell_is_loaded_via_state and hasattr(symspell_from_state, 'words') else 0
    supabase_client_ok = getattr(request.app.state, 'supabase_client') is not None
    
    status_detail = ""
    if lifespan_run:
        if critical_loaded and supabase_client_ok:
            status_detail = " - Durum: Aktif, Temel NLU Kaynakları ve Supabase İstemcisi Yüklendi."
            if symspell_is_loaded_via_state: status_detail += f" SymSpell Aktif (Kelime Sayısı: {symspell_word_count_via_state})."
            else: status_detail += " SymSpell ETKİN DEĞİL veya YÜKLENEMEDİ."
        elif not supabase_client_ok:
             status_detail = " - Durum: Kısmen Aktif, NLU Kaynakları Yüklendi ama SUPABASE İSTEMCİSİ YÜKLENEMEDİ."
        else: status_detail = " - Durum: Etkin Değil, KRİTİK NLU KAYNAKLARI YÜKLENEMEDİ."
    else: status_detail = " - Durum: Etkin Değil, LIFESPAN ÇALIŞMADI veya erken başarısız oldu."
    
    return {
        "message": f"{api_version_message}{status_detail}",
        "app_state_lifespan_executed": lifespan_run,
        "app_state_critical_resources_loaded": critical_loaded,
        "app_state_supabase_client_loaded": supabase_client_ok,
        "app_state_symspell_loaded": symspell_is_loaded_via_state,
        "app_state_symspell_word_count": symspell_word_count_via_state,
        "turkish_frequency_dictionary_path_exists": TURKISH_FREQUENCY_DICTIONARY_PATH.exists(),
        "nlu_model_path_exists": MODEL_PATH.exists()
    }

# Lokal geliştirme için:
# if __name__ == "__main__":
#     import uvicorn
#     logger.info(f"Uvicorn ile lokal geliştirme sunucusu başlatılıyor. Temel dizin: {BASE_DIR}")
#     logger.info(f"NLU Model yolu: {MODEL_PATH}")
#     logger.info(f"Frekans sözlüğü yolu: {TURKISH_FREQUENCY_DICTIONARY_PATH}")
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
