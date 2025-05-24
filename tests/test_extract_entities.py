import sys
import os
from pathlib import Path
import pytest
import logging # logging modülü import edildi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from main import (
        extract_simple_entities,
        preprocess_query_for_nlu,
        _preprocess_text_for_matching, # _preprocess_text_for_matching'i de import etmeliyiz
        simulated_business_data,
        TURKISH_FREQUENCY_DICTIONARY_PATH as main_turkish_freq_path, # main'deki yolu kullanmak için
        correct_spelling
    )
    import main # main modülünü import et ki global değişkenlerine atama yapabilelim
    import zeyrek
    from symspellpy import SymSpell, Verbosity # Verbosity eklendi
except ImportError as e:
    print(f"Import Hatası (test_extract_entities.py): {e}\nPython Path: {sys.path}")
    raise

# Logging'i test dosyasında da aktif edelim
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Testler için DEBUG seviyesini ayarlayabiliriz

@pytest.fixture(scope="module", autouse=True)
def setup_module_resources_for_entity_tests(request):
    """
    Bu fixture, test_extract_entities.py içindeki testler çalışmadan önce
    main.py'deki global kaynakları (morphology, sym_spell, db_map) yükler.
    Bu, test_api.py'deki TestClient'ın tetiklediği lifespan'dan bağımsızdır ve
    main.py'deki fonksiyonların unit testleri için gereklidir.
    """
    logger.info("\n--- (test_extract_entities.py setup_module_resources) Kaynaklar yükleniyor/kontrol ediliyor ---")

    # 1. Morfoloji Analizörünü Yükle (Zeyrek)
    if main.morphology is None:
        logger.info("main.morphology None. Testler için Zeyrek manuel olarak yükleniyor...")
        try:
            main.morphology = zeyrek.MorphAnalyzer()
            logger.info("main.morphology testler için Zeyrek ile dolduruldu.")
        except Exception as e:
            logger.error(f"HATA (setup): Testler için Zeyrek yüklenirken: {e}", exc_info=True)
            pytest.skip("Zeyrek yüklenemedi, varlık çıkarma testleri atlanıyor.", allow_module_level=True)
            return # Fixture'dan çık
    else:
        logger.info(f"main.morphology zaten yüklü. Tip: {type(main.morphology)}")

    # 2. SymSpell Yazım Denetleyicisini Yükle
    if main.sym_spell is None:
        logger.info("main.sym_spell None. Testler için SymSpell manuel olarak yükleniyor...")
        try:
            temp_sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            # main.py'den import edilen TURKISH_FREQUENCY_DICTIONARY_PATH'ı kullanıyoruz
            # Bu değişkenin main.py'de doğru tanımlandığından emin olun.
            # Eğer main.py'de bu değişken yoksa, doğrudan yolu burada belirtin veya
            # main.py'deki global değişkeni kullanın (örn: main.TURKISH_FREQUENCY_DICTIONARY_PATH)
            dictionary_path = main_turkish_freq_path # main.py'den import edilen yolu kullan
            logger.info(f"SymSpell dictionary path being used for tests: {dictionary_path.resolve()}")
            if dictionary_path.exists() and dictionary_path.is_file():
                if temp_sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1, separator="\t", encoding="utf-8"):
                    main.sym_spell = temp_sym_spell # main.py'deki global değişkene ata
                    logger.info(f"SymSpell testler için yüklendi ve main.sym_spell'e atandı. Word count: {len(main.sym_spell.words) if main.sym_spell and hasattr(main.sym_spell, 'words') else 'N/A'}")
                else:
                    logger.error(f"HATA: SymSpell dictionary could not be loaded from {dictionary_path} (load_dictionary returned False).")
                    # SymSpell yüklenemezse bile testlere devam et, correct_spelling bunu handle eder (uyarı verir).
            else:
                logger.error(f"HATA: SymSpell dictionary file NOT FOUND at {dictionary_path}.")
        except Exception as e:
            logger.error(f"HATA (setup): Testler için SymSpell yüklenirken: {e}", exc_info=True)
    else:
        logger.info(f"main.sym_spell zaten yüklü. Word count: {len(main.sym_spell.words) if main.sym_spell and hasattr(main.sym_spell, 'words') else 'N/A'}")

    # 3. DB Ürün Haritasını Oluştur (main.db_product_original_to_processed_map üzerinde çalış)
    # Bu harita, _preprocess_text_for_matching fonksiyonunu kullanır, bu da global main.morphology'ye bağlıdır.
    if main.morphology: # Sadece morfoloji yüklendiyse haritayı oluştur
        if not main.db_product_original_to_processed_map: # Eğer harita boşsa
            logger.info("DB product map (main.db_product_original_to_processed_map) testler için dolduruluyor...")
            try:
                for original_name_key in simulated_business_data["products"].keys():
                    # _preprocess_text_for_matching'i main'den import ettiğimizle kullanalım.
                    # Bu fonksiyonun global main.morphology'yi kullandığından emin olun.
                    processed_name_value = _preprocess_text_for_matching(original_name_key)
                    main.db_product_original_to_processed_map[original_name_key] = processed_name_value
                logger.info(f"DB product map (main.db_product_original_to_processed_map) dolduruldu: {len(main.db_product_original_to_processed_map)} öğe.")
                logger.debug(f"DB Product Map Content (Fixture): {main.db_product_original_to_processed_map}")
            except Exception as e:
                logger.error(f"HATA (setup): DB product map doldurulurken: {e}", exc_info=True)
                pytest.skip("DB product map doldurulamadı.", allow_module_level=True)
                return
        else:
            logger.info("main.db_product_original_to_processed_map zaten dolu.")
    elif not main.morphology:
         logger.warning("Morfoloji yüklenmediği için DB product map doldurulamadı.")
         pytest.skip("Kritik kaynak (Zeyrek) yüklenemedi, DB map oluşturulamıyor.", allow_module_level=True)
         return

    logger.info("--- (test_extract_entities.py setup_module_resources) Kaynak yükleme tamamlandı ---")


@pytest.mark.parametrize(
    "original_query, processed_query_override, intent, expected_product, expected_size, expected_is_generic, expected_generic_options_count",
    [
        # Mevcut başarılı testler
        ("Keten Pantolon fiyatı nedir", None, "fiyat_sorgulama", "Keten Pantolon", None, False, 0),
        ("İpek Gömleğin fiyatı ne kadar", None, "fiyat_sorgulama", "İpek Gömlek", None, False, 0),
        ("ktene pantolon fiyat", None, "fiyat_sorgulama", "Keten Pantolon", None, False, 0), # SymSpell düzeltmeli
        ("KETEN PANTOON FİYAT", None, "fiyat_sorgulama", "Keten Pantolon", None, False, 0), # SymSpell düzeltmeli
        ("42 si var mı?", None, "stok_sorgulama", None, "42", False, 0),
        ("bunun fiyatı nedir", None, "fiyat_sorgulama", None, None, False, 0), # Anaphoric, ürün yok
        ("kolay gelsin, Deri Ceketin malzemesi nedir acaba?", None, "ürün_malzeme_sorma", "Deri Ceket", None, False, 0),
        ("pantolon fiyatları", None, "fiyat_sorgulama", "pantolon", None, True, 2),
        ("bu ürün var mı", None, "stok_sorgulama", None, None, False, 0), # Anaphoric, ürün yok

        # Ekstra test senaryoları
        ("Deri Ceket S beden", None, "stok_sorgulama", "Deri Ceket", "S", False, 0),
        ("kot pantolon 30 beden var mı", None, "stok_sorgulama", "Kot Pantolon", "30", False, 0),
        ("ipek gömlek small beden", None, "stok_sorgulama", "İpek Gömlek", "S", False, 0),
        ("gömlek var mı", None, "stok_sorgulama", "gömlek", None, True, 1),
        ("ceketler ne kadar", None, "fiyat_sorgulama", "ceket", None, True, 1),
        ("kırmızı keten pantolon", None, "ürün_bilgisi_sorma", "Keten Pantolon", None, False, 0), # Renk çıkarımı yok, ürün doğru olmalı
        ("bu kabanın fiyatı", None, "fiyat_sorgulama", None, None, False, 0), # Anaphoric, ürün yok (kaban DB'de yok)
        ("İpek Gömlek fiyati ne kadr", None, "fiyat_sorgulama", "İpek Gömlek", None, False, 0), # SymSpell düzeltmeli
        ("Deri Ceket XL kaldı mı", None, "stok_sorgulama", "Deri Ceket", "XL", False, 0),
        ("Keten Pantolon kumaşı nedr", None, "ürün_malzeme_sorma", "Keten Pantolon", None, False, 0), # SymSpell düzeltmeli
        ("sadece pantolon", None, "ürün_bilgisi_sorma", "pantolon", None, True, 2),
        ("kot pantolon özellikleri", None, "ürün_bilgisi_sorma", "Kot Pantolon", None, False, 0),
        ("PANTOLON", None, "ürün_bilgisi_sorma", "pantolon", None, True, 2),
        ("KOT PANTOLON", None, "ürün_bilgisi_sorma", "Kot Pantolon", None, False, 0),
        ("deri ceketten var mı?", "deri ceket var mı", "stok_sorgulama", "Deri Ceket", None, False, 0), # Lemmatize edilmiş hali test ediliyor

        # Yazım denetimi testleri (SymSpell'in doğru çalışmasına bağlı)
        ("Keten pntolon fiyatı", None, "fiyat_sorgulama", "Keten Pantolon", None, False, 0),
        ("İpek gömlek s beden", None, "stok_sorgulama", "İpek Gömlek", "S", False, 0),
        ("İpek gömlek smal beden", None, "stok_sorgulama", "İpek Gömlek", "S", False, 0),
        ("Deri jeket özellikleri", None, "ürün_bilgisi_sorma", "Deri Ceket", None, False, 0),
        ("Kot pantalon M", None, "stok_sorgulama", "Kot Pantolon", "M", False, 0),
        ("Yün kazak fiyati", None, "fiyat_sorgulama", None, None, False, 0), # Yün kazak DB'de yok, ürün None olmalı
        ("Keten pantolon XL bedn", None, "stok_sorgulama", "Keten Pantolon", "XL", False, 0),
    ]
)
def test_extract_entities_parametrized(original_query, processed_query_override, intent, expected_product, expected_size, expected_is_generic, expected_generic_options_count):
    if main.morphology is None: # Fixture'ın yüklendiğinden emin ol
        pytest.skip("Morfoloji yüklenemedi (test_extract_entities_parametrized), bu test atlanıyor.", allow_module_level=True)
    if main.sym_spell is None: # Symspell de kontrol edilebilir, ancak correct_spelling zaten uyarı veriyor.
        logger.warning(f"UYARI (test_extract_entities_parametrized): main.sym_spell None. '{original_query}' için yazım denetimi yapılmayacak/eksik olabilir.")

    # Test için kullanılacak spell-checked orijinal sorgu
    # Bu, extract_simple_entities'in ilk parametresi için önemlidir.
    spell_checked_original_for_entities = correct_spelling(original_query)

    # Test için kullanılacak tam işlenmiş sorgu (spell-check + lemmatization)
    # Bu, extract_simple_entities'in ikinci parametresi için önemlidir.
    processed_query_for_entities = processed_query_override if processed_query_override else preprocess_query_for_nlu(original_query)

    entities = extract_simple_entities(spell_checked_original_for_entities, processed_query_for_entities, intent)

    logger.debug(f"DEBUG ENTITY TEST:\n"
                 f"  Original Query: '{original_query}'\n"
                 f"  Spell-checked Original for Entities: '{spell_checked_original_for_entities}'\n"
                 f"  Processed Query for NLU (Lemmatized): '{processed_query_for_entities}'\n"
                 f"  Intent: '{intent}'\n"
                 f"  EXTRACTED Entities: {entities}\n"
                 f"  EXPECTED Product: '{expected_product}', Size: '{expected_size}', IsGeneric: {expected_is_generic}\n"
                 f"--------------------------------------------------")

    assert entities.get("product") == expected_product, \
        f"Query: '{original_query}'. Expected product '{expected_product}', got '{entities.get('product')}'"
    assert entities.get("size") == expected_size, \
        f"Query: '{original_query}'. Expected size '{expected_size}', got '{entities.get('size')}'"
    assert entities.get("is_generic_product_term") is expected_is_generic, \
        f"Query: '{original_query}'. Expected is_generic_product_term to be {expected_is_generic}, got '{entities.get('is_generic_product_term')}'"

    if expected_is_generic:
        assert entities.get("generic_term_options") is not None, f"Query: '{original_query}'. Expected generic_term_options not to be None for a generic product."
        assert len(entities.get("generic_term_options", [])) == expected_generic_options_count, \
             f"Query: '{original_query}'. Expected {expected_generic_options_count} generic options, got {len(entities.get('generic_term_options', []))}. Options: {entities.get('generic_term_options')}"
        # Teste özel kategori-ürün eşleşmelerini kontrol et
        if expected_product == "pantolon":
             assert "Keten Pantolon" in entities.get("generic_term_options", [])
             assert "Kot Pantolon" in entities.get("generic_term_options", [])
        elif expected_product == "gömlek":
             assert "İpek Gömlek" in entities.get("generic_term_options", [])
        elif expected_product == "ceket":
             assert "Deri Ceket" in entities.get("generic_term_options", [])


# Bireysel testler, parametrik testler daha kapsamlı olduğu için yorum satırına alınabilir
# veya spesifik hata ayıklama için tutulabilir.
def test_urun_adi_basit_fiyat_sorgusu():
    original_query = "Keten Pantolon fiyatı nedir"
    spell_checked_original = correct_spelling(original_query)
    processed_query = preprocess_query_for_nlu(original_query)
    intent = "fiyat_sorgulama"
    entities = extract_simple_entities(spell_checked_original, processed_query, intent)
    assert entities.get("product") == "Keten Pantolon"
    assert entities.get("size") is None

def test_urun_adi_ek_ile_fiyat_sorgusu():
    original_query = "İpek Gömleğin fiyatı ne kadar"
    spell_checked_original = correct_spelling(original_query)
    processed_query = preprocess_query_for_nlu(original_query)
    intent = "fiyat_sorgulama"
    entities = extract_simple_entities(spell_checked_original, processed_query, intent)
    assert entities.get("product") == "İpek Gömlek"

def test_hatali_yazim_urun_fiyat_sorgusu():
    original_query = "ktene pantolon fiyat"
    spell_checked_original = correct_spelling(original_query)
    processed_query = preprocess_query_for_nlu(original_query)
    intent = "fiyat_sorgulama"
    entities = extract_simple_entities(spell_checked_original, processed_query, intent)
    assert entities.get("product") == "Keten Pantolon", f"Beklenen 'Keten Pantolon', alınan '{entities.get('product')}'"

def test_buyuk_harf_ve_hatali_yazim_urun_fiyat_sorgusu():
    original_query = "KETEN PANTOON FİYAT"
    spell_checked_original = correct_spelling(original_query)
    processed_query = preprocess_query_for_nlu(original_query)
    intent = "fiyat_sorgulama"
    entities = extract_simple_entities(spell_checked_original, processed_query, intent)
    assert entities.get("product") == "Keten Pantolon"

def test_kisa_sorgu_urun_yok_beden_var():
    original_query = "42 si var mı?"
    spell_checked_original = correct_spelling(original_query)
    processed_query = preprocess_query_for_nlu(original_query)
    intent = "stok_sorgulama"
    entities = extract_simple_entities(spell_checked_original, processed_query, intent)
    assert entities.get("product") is None
    assert entities.get("size") == "42"

def test_zamir_ile_kisa_sorgu_urun_yok():
    original_query = "bunun fiyatı nedir"
    spell_checked_original = correct_spelling(original_query)
    processed_query = preprocess_query_for_nlu(original_query)
    intent = "fiyat_sorgulama"
    entities = extract_simple_entities(spell_checked_original, processed_query, intent)
    assert entities.get("product") is None

def test_dolgu_ifadeli_uzun_sorgu_urun_cikarma():
    original_query = "kolay gelsin, Deri Ceketin malzemesi nedir acaba?"
    spell_checked_original = correct_spelling(original_query)
    processed_query = preprocess_query_for_nlu(original_query)
    intent = "ürün_malzeme_sorma"
    entities = extract_simple_entities(spell_checked_original, processed_query, intent)
    assert entities.get("product") == "Deri Ceket"

def test_genel_urun_terimi_ile_sorgu():
    original_query = "pantolon fiyatları"
    spell_checked_original = correct_spelling(original_query)
    processed_query = preprocess_query_for_nlu(original_query)
    intent = "fiyat_sorgulama"
    entities = extract_simple_entities(spell_checked_original, processed_query, intent)
    assert entities.get("product") == "pantolon", f"Beklenen 'pantolon', alınan '{entities.get('product')}'"
    assert entities.get("is_generic_product_term") is True
    assert "Keten Pantolon" in entities.get("generic_term_options", [])

def test_bu_urun_var_mi_sorgusu():
    original_query = "bu ürün var mı"
    spell_checked_original = correct_spelling(original_query)
    processed_query = preprocess_query_for_nlu(original_query)
    intent = "stok_sorgulama"
    entities = extract_simple_entities(spell_checked_original, processed_query, intent)
    assert entities.get("product") is None
