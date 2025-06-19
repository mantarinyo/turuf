import sys
from fastapi.testclient import TestClient
from pathlib import Path
import pytest
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

from main import app

def check_resources_loaded(client):
    lifespan_executed = getattr(client.app.state, 'lifespan_was_executed', False)
    resources_loaded = getattr(client.app.state, 'critical_resources_loaded', False)
    
    if not lifespan_executed:
        pytest.skip("Skipping API test as lifespan did not seem to execute (checked via client.app.state).")
    if not resources_loaded:
        pytest.skip("Skipping API test as critical resources are not loaded (checked via client.app.state).")

@pytest.fixture(scope="module")
def client_fixture():
    logger.info("--- test_api.py: client_fixture SETUP ---")
    with TestClient(app) as client:
        try:
            initial_response = client.get("/")
            logger.info(f"Initial GET / response status: {initial_response.status_code}")
            if initial_response.status_code != 200:
                 logger.error(f"Initial GET / failed with status {initial_response.status_code}, content: {initial_response.text}")
        except Exception as e:
            logger.error(f"Error during initial GET / in client_fixture: {e}", exc_info=True)
        yield client
    logger.info("--- test_api.py: client_fixture TEARDOWN ---")

def test_read_root(client_fixture):
    client = client_fixture
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    print(f"DEBUG test_read_root data: {data}")
    assert "message" in data
    
    expected_api_version = "NLU API (v23.8 - SymSpell Guardrails & Enhanced Entity Logging)"
    assert expected_api_version in data["message"]
    assert data.get("app_state_lifespan_executed") is True, f"app.state.lifespan_was_executed True olmalı. Data: {data}"
    assert data.get("app_state_critical_resources_loaded") is True, f"app.state.critical_resources_loaded True olmalı. Data: {data}"
    
    if data.get("app_state_symspell_loaded") is True:
        assert "SymSpell Active" in data["message"], f"Mesaj 'SymSpell Active' içermeli. Alınan: '{data['message']}'"
        assert f"Word Count: {data.get('global_sym_spell_word_count')}" in data["message"]
    else:
        assert "SymSpell INACTIVE" in data["message"] or "SymSpell FAILED" in data["message"]

@pytest.mark.parametrize(
    "query, session_id, expected_intent, expected_product, expected_size, partial_response_keyword, should_ask_for_clarification",
    [
        # --- Fiyat Sorgulama ---
        ("fiyatını nedir", None, "fiyat_sorgulama", None, None, "Hangi ürünün fiyatını soruyorsunuz", False),
        ("ücreti ne kadar acaba", None, "fiyat_sorgulama", None, None, "Hangi ürünün fiyatını soruyorsunuz", False),
        ("kaç para bu", None, "fiyat_sorgulama", None, None, "Hangi ürünün fiyatını soruyorsunuz", False),
        ("bu modelin fiyatını öğrenebilir miyim", None, "fiyat_sorgulama", None, None, "Hangi ürünün fiyatını soruyorsunuz", False),
        ("ne kadar?", None, "fiyat_sorgulama", None, None, "Hangi ürünün fiyatını soruyorsunuz", False),
        ("bu kaç tl", None, "fiyat_sorgulama", None, None, "Hangi ürünün fiyatını soruyorsunuz", False),
        ("fiyay nedir", None, "fiyat_sorgulama", None, None, "Hangi ürünün fiyatını soruyorsunuz", False), # Typo
        ("fyt ne", None, "fiyat_sorgulama", None, None, "Hangi ürünün fiyatını soruyorsunuz", False), # Typo
        ("Keten Pantolon fiyat", None, "fiyat_sorgulama", "Keten Pantolon", None, "850 TL", False),
        ("İpek Gömlek fiyati ne kadr", None, "fiyat_sorgulama", "İpek Gömlek", None, "1250 TL", False), # Typo
        ("Deri Ceket kça para", None, "fiyat_sorgulama", "Deri Ceket", None, "3200 TL", False), # Typo
        ("pantolon fiyatları", None, "fiyat_sorgulama", None, None, "'Pantolon' ile ilgili birkaç ürünümüz var", True),

        # --- Selamlama ---
        ("merhaba nasılsınız", None, "selamlama", None, None, "Merhaba! Size nasıl yardımcı olabilirim", False),
        ("selamlar iyi çalışmalar", None, "selamlama", None, None, "Merhaba! Size nasıl yardımcı olabilirim", False),
        ("günaydın kolay gelsin", None, "selamlama", None, None, "Merhaba! Size nasıl yardımcı olabilirim", False),
        ("mrb", None, "selamlama", None, None, "Merhaba! Size nasıl yardımcı olabilirim", False), 
        ("slm", None, "selamlama", None, None, "Merhaba! Size nasıl yardımcı olabilirim", False), 
        ("merhba", None, "selamlama", None, None, "Merhaba! Size nasıl yardımcı olabilirim", False), 
        ("gunaydn", None, "selamlama", None, None, "Merhaba! Size nasıl yardımcı olabilirim", False), 

        # --- Stok Sorgulama ---
        ("bunun 42 si var mı", None, "stok_sorgulama", None, "42", "Hangi ürünün stok durumunu soruyorsunuz", False), 
        ("medium bedeni mevcut mu", None, "stok_sorgulama", None, "M", "Hangi ürünün stok durumunu soruyorsunuz", False), 
        ("Keten Pantolon L bedeni mevcutmu", None, "stok_sorgulama", "Keten Pantolon", "L", "'Keten Pantolon' L bedeni için", False), 
        ("İpek Gömlek stokta varmi", None, "stok_sorgulama", "İpek Gömlek", None, "'İpek Gömlek' ürününün tüm beden ve stok bilgileri", False), 
        ("Deri Ceket XL kaldı mı", None, "stok_sorgulama", "Deri Ceket", "XL", "'Deri Ceket' XL bedeni için", False),
        ("pantolon stok", None, "stok_sorgulama", None, None, "'Pantolon' ile ilgili birkaç ürünümüz var", True),

        # --- İade ve Değişim ---
        ("iade var mı acaba", None, "iade_sorgulama", None, None, "Genel iade politikamız", False),
        ("ürünü değiştirebilir miyim", None, "iade_sorgulama", None, None, "Genel iade politikamız", False),
        ("para iadesi mümkünmü", None, "iade_sorgulama", None, None, "Genel iade politikamız", False), 
        ("Keten Pantolonu iade etmek istiyorum", None, "iade_sorgulama", "Keten Pantolon", None, "'Keten Pantolon' ile ilgili iade politikamız", False),

        # --- Scenario Group 1: Short Product Name Queries (Updated Expectations) ---
        ("İpek Gömlek", None, "ürün_bilgisi_sorma", "İpek Gömlek", None, "%100 saf ipekten", False),
        ("Keten Pantolon", None, "ürün_bilgisi_sorma", "Keten Pantolon", None, "%100 doğal ketenden", False),
        ("deri ceket", None, "ürün_bilgisi_sorma", "Deri Ceket", None, "Hakiki kuzu derisinden", False),
        ("ipek gömleği", None, "ürün_bilgisi_sorma", "İpek Gömlek", None, "%100 saf ipekten", False), 
        ("kot pantolonlar", None, "ürün_bilgisi_sorma", "Kot Pantolon", None, "dayanıklı kot pantolon", False), 
        ("sadece ipek gömlek", None, "ürün_bilgisi_sorma", "İpek Gömlek", None, "%100 saf ipekten", False),
        # ("Deri Ceket", None, "ürün_bilgisi_sorma", "Deri Ceket", None, "Hakiki kuzu derisinden", False), # Bu "deri ceket" ile aynı, birini seçebilirsiniz.

        # --- Ürün Malzeme/İçerik ---
        ("bu pantolonun kumaşı nedir", None, "ürün_malzeme_sorma", None, None, "Hangi ürünün malzeme/içerik bilgisini soruyorsunuz", False), 
        ("Deri Ceketin içeriğinde ne var", None, "ürün_malzeme_sorma", "Deri Ceket", None, "Hakiki Kuzu Derisi", False),
        ("İpek Gömlek malzemesi ne", None, "ürün_malzeme_sorma", "İpek Gömlek", None, "%100 İpek", False),
        ("Keten Pantolon kumas ne", None, "ürün_malzeme_sorma", "Keten Pantolon", None, "%100 Keten", False), 

        # --- Genel Ürün Bilgisi ---
        ("bunun özellikleri nelerdir", None, "ürün_bilgisi_sorma", None, None, "Hangi ürün hakkında bilgi almak istiyorsunuz", False), 
        ("Keten Pantolon hakkında daha fazla detay", None, "ürün_bilgisi_sorma", "Keten Pantolon", None, "Yüksek kaliteli, %100 doğal keten", False),
        ("İpek Gömlek urun bilgisi", None, "ürün_bilgisi_sorma", "İpek Gömlek", None, "%100 saf ipekten", False), 

        # --- İletişim Bilgileri ---
        ("tel no var mı", None, "tel_no_sorma", None, None, "Telefon: 0555", False),
        ("whatsapp no", None, "tel_no_sorma", None, None, "WhatsApp: 0555", False),
        ("mail adresiniz nedir", None, "tel_no_sorma", None, None, "info@mantarinyo-butik.com", False),

        # --- Lokasyon ---
        ("konumunuz nerede acaba tam olarak", None, "lokasyon_sorma", None, None, "Moda Caddesi No:1", False),
        ("adresiniz neydi", None, "lokasyon_sorma", None, None, "Moda Caddesi No:1", False),
        ("dükkan nerde abi", None, "lokasyon_sorma", None, None, "Moda Caddesi No:1", False), 

        # --- Kargo Bilgisi ---
        ("kargo ücretli mi", None, "kargo_bilgisi_sorma", None, None, "Siparişleriniz Yurtiçi Kargo", False),
        ("kargonuz kaç günde teslim ediyor", None, "kargo_bilgisi_sorma", None, None, "2-3 iş gününde", False),
        ("kargo nekadar tutar", None, "kargo_bilgisi_sorma", None, None, "Kargo ücreti 50 TL", False), 

        # --- Çalışma Saatleri ---
        ("bugün kaça kadar açıksınız", None, "calisma_saatleri_sorma", None, None, "Haftaiçi: 09:00-19:00", False),
        ("pazar açıkmı mekan", None, "calisma_saatleri_sorma", None, None, "Pazar: Kapalı", False), 

        # --- Ödeme Yöntemleri ---
        ("kredi kartıyla ödeme yapabilir miyim", None, "odeme_yontemleri_sorma", None, None, "Kredi Kartı (Visa, Mastercard, Amex)", False),
        ("taksit imkanı var mı", None, "odeme_yontemleri_sorma", None, None, "Kredi Kartı", False), 
        ("kapıda ödeme kabul ediyor musunuz", None, "odeme_yontemleri_sorma", None, None, "Havale/EFT kabul edilmektedir", False), 

        # --- Teşekkür ---
        ("teşekkür ederim bilgilendirme için", None, "tesekkur", None, None, "Rica ederim!", False),
        ("sağ olun yardımınız için", None, "tesekkur", None, None, "Rica ederim!", False),
        ("tşk", None, "tesekkur", None, None, "Rica ederim!", False), 
        ("eyw", None, "tesekkur", None, None, "Rica ederim!", False), 

        # --- Kapsam Dışı ---
        ("bugün hava nasıl olacak", None, "kapsam_disi", None, None, "Üzgünüm, bu konuda yardımcı olamıyorum", False),
        ("bana bir fıkra anlat", None, "kapsam_disi", None, None, "Üzgünüm, bu konuda yardımcı olamıyorum", False),
        ("sdfgsdfg", None, "kapsam_disi", None, None, "Üzgünüm, bu konuda yardımcı olamıyorum", False), 

        ("Keten pantalon fiyati nedir?", None, "fiyat_sorgulama", "Keten Pantolon", None, "'Keten Pantolon' ürününün fiyatı", False),
        ("İpek gömlek bednleri var mı?", None, "stok_sorgulama", "İpek Gömlek", None, "'İpek Gömlek' ürününün tüm beden ve stok bilgileri", False),
        ("Deri ceket iade koşullaeı nelerdir?", None, "iade_sorgulama", "Deri Ceket", None, "'Deri Ceket' ile ilgili iade politikamız", False),
        ("Magazanız nerede?", None, "lokasyon_sorma", None, None, "Adresimiz: Moda Caddesi", False),
        ("Calısma saatleriniz nedir?", None, "calisma_saatleri_sorma", None, None, "Çalışma saatlerimiz:", False),
        ("Bu ürnün fiyati ne kadar?", None, "fiyat_sorgulama", None, None, "Hangi ürünün fiyatını soruyorsunuz", False),
        ("Keten pantolonun kırmzı rengi var mı?", None, "stok_sorgulama", "Keten Pantolon", None, "'Keten Pantolon' ürününün tüm beden ve stok bilgileri", False),
    ]
)
def test_various_queries_api(client_fixture, query, session_id, expected_intent, expected_product, expected_size, partial_response_keyword, should_ask_for_clarification):
    client = client_fixture
    check_resources_loaded(client)
    
    response = client.post("/process_query/", json={"query": query, "session_id": session_id}) 
    assert response.status_code == 200
    data = response.json()
    print(f"DEBUG API Test. Query: '{query}'. Response: {data}")

    assert data.get("detected_intent") == expected_intent, \
        f"Query: '{query}'. Expected intent '{expected_intent}', got '{data.get('detected_intent')}'"
    
    if expected_product:
        assert data.get("resolved_product") == expected_product, \
            f"Query: '{query}'. Expected product '{expected_product}', got '{data.get('resolved_product')}'"
    elif not expected_product and not should_ask_for_clarification: 
        assert data.get("resolved_product") is None, \
            f"Query: '{query}'. Expected no resolved_product (got '{data.get('resolved_product')}') as no specific product was expected and clarification was not expected."

    if expected_size:
        assert data.get("resolved_size") == expected_size, \
            f"Query: '{query}'. Expected size '{expected_size}', got '{data.get('resolved_size')}'"

    assert data.get("ask_for_clarification") is should_ask_for_clarification, \
        f"Query: '{query}'. Expected ask_for_clarification to be {should_ask_for_clarification}, got '{data.get('ask_for_clarification')}'"

    if partial_response_keyword:
        assert partial_response_keyword.lower() in data.get("bot_response", "").lower(), \
            f"Query: '{query}'. Expected bot_response to contain '{partial_response_keyword}', got '{data.get('bot_response')}'"

def test_selamlama_basit_api(client_fixture):
    client = client_fixture
    check_resources_loaded(client)
    response = client.post("/process_query/", json={"query": "merhaba", "session_id": None})
    assert response.status_code == 200
    data = response.json()
    print(f"DEBUG test_selamlama_basit_api data: {data}")
    # Düzeltme: 'regex_specific_selamlama' ve 'regex_kept_selamlama' kabul edilebilir metodlar listesine eklendi
    assert data.get("nlu_method") in ["regex", "slm_fasttext_override", "slm_fasttext", "regex_specific_selamlama", "regex_kept_selamlama"]
    final_intent = data.get("detected_intent")
    assert final_intent == "selamlama"
    assert "Merhaba! Size nasıl yardımcı olabilirim?" in data.get("bot_response", "") 
    assert data.get("resolved_product") is None
    assert data.get("session_id") is not None

def test_urun_fiyat_sorgulama_net_api(client_fixture):
    client = client_fixture
    check_resources_loaded(client)
    response = client.post("/process_query/", json={"query": "Keten Pantolon fiyatı nedir", "session_id": None})
    assert response.status_code == 200
    data = response.json()
    print(f"DEBUG test_urun_fiyat_sorgulama_net_api data: {data}")
    assert data.get("resolved_product") == "Keten Pantolon"
    assert "850 TL" in data.get("bot_response", "")
    assert data.get("ask_for_clarification") is False

def test_baglam_bosken_urunsuz_fiyat_sorgulama_api(client_fixture):
    client = client_fixture
    check_resources_loaded(client)
    response = client.post("/process_query/", json={"query": "fiyatı ne kadar", "session_id": None})
    assert response.status_code == 200
    data = response.json()
    print(f"DEBUG test_baglam_bosken_urunsuz_fiyat_sorgulama_api data: {data}")
    final_intent = data.get("detected_intent")
    assert final_intent == "fiyat_sorgulama"
    assert data.get("resolved_product") is None
    assert "Hangi ürünün fiyatını soruyorsunuz?" in data.get("bot_response", "")
    assert data.get("ask_for_clarification") is False

def test_baglam_kullanimi_beden_sorgusu_api(client_fixture):
    client = client_fixture
    check_resources_loaded(client)

    response1 = client.post("/process_query/", json={"query": "İpek Gömlek hakkında bilgi", "session_id": "test_session_context_1"})
    assert response1.status_code == 200
    data1 = response1.json()
    session_id = data1.get("session_id")
    assert session_id == "test_session_context_1" 
    assert data1.get("resolved_product") == "İpek Gömlek"

    response2 = client.post("/process_query/", json={"query": "M bedeni var mı?", "session_id": session_id})
    assert response2.status_code == 200
    data2 = response2.json()
    print(f"DEBUG test_baglam_kullanimi_beden_sorgusu_api data2: {data2}")

    final_intent2 = data2.get("detected_intent")
    assert final_intent2 == "stok_sorgulama"
    assert data2.get("resolved_product") == "İpek Gömlek" 
    assert data2.get("resolved_size") == "M"
    assert "İpek Gömlek" in data2.get("bot_response", "")
    assert "M bedeni için" in data2.get("bot_response", "") or "'M' bedeni için" in data2.get("bot_response", "")
    assert data2.get("previous_query_in_session") == "İpek Gömlek hakkında bilgi"
    assert data2.get("ask_for_clarification") is False

def test_kapsam_disi_sorgu_api(client_fixture):
    client = client_fixture
    check_resources_loaded(client)
    response = client.post("/process_query/", json={"query": "bana bir fıkra anlat", "session_id": None})
    assert response.status_code == 200
    data = response.json()
    print(f"DEBUG test_kapsam_disi_sorgu_api data: {data}")
    final_intent = data.get("detected_intent")
    assert final_intent == "kapsam_disi"
    assert "Üzgünüm, bu konuda yardımcı olamıyorum." in data.get("bot_response", "")
    assert data.get("ask_for_clarification") is False

def test_yazim_hatali_urun_sorgusu_api(client_fixture): 
    client = client_fixture
    check_resources_loaded(client)
    response = client.post("/process_query/", json={"query": "ktene pantolon fiyat", "session_id": None})
    assert response.status_code == 200
    data = response.json()
    print(f"DEBUG test_yazim_hatali_urun_sorgusu_api data: {data}")
    assert data.get("resolved_product") == "Keten Pantolon" 
    assert "850 TL" in data.get("bot_response", "")
    assert data.get("ask_for_clarification") is False

# Yeni eklenen bağlamsal testler
@pytest.mark.asyncio
async def test_contextual_product_clarification_then_price(client_fixture):
    client = client_fixture
    check_resources_loaded(client)
    session_id = "test_context_price_1"

    # 1. Kullanıcı genel bir ürün kategorisiyle fiyat sorar
    response1 = client.post("/process_query/", json={"query": "gömlek fiyatları", "session_id": session_id})
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1.get("detected_intent") == "fiyat_sorgulama"
    assert data1.get("ask_for_clarification") is True
    assert "'İpek Gömlek'" in data1.get("clarification_options", [])
    assert "Hangisini sormuştunuz?" in data1.get("bot_response")

    # 2. Kullanıcı sadece ürün adını belirterek yanıt verir
    response2 = client.post("/process_query/", json={"query": "İpek Gömlek", "session_id": session_id})
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2.get("detected_intent") == "fiyat_sorgulama", \
        f"Beklenen niyet 'fiyat_sorgulama', alınan '{data2.get('detected_intent')}'"
    assert data2.get("resolved_product") == "İpek Gömlek", \
        f"Beklenen ürün 'İpek Gömlek', alınan '{data2.get('resolved_product')}'"
    assert "1250 TL" in data2.get("bot_response", ""), \
        f"Bot yanıtı '1250 TL' içermeli, alınan: '{data2.get('bot_response')}'"
    assert data2.get("ask_for_clarification") is False

@pytest.mark.asyncio
async def test_contextual_product_clarification_then_price(client_fixture): # client_fixture parametresini ekledim
    client = client_fixture # client_fixture'ı kullan
    check_resources_loaded(client)
    session_id = "test_context_price_1"

    # 1. Kullanıcı genel bir ürün kategorisiyle fiyat sorar
    response1 = client.post("/process_query/", json={"query": "gömlek fiyatları", "session_id": session_id})
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1.get("detected_intent") == "fiyat_sorgulama"
    assert data1.get("ask_for_clarification") is True
    # ---- DÜZELTİLMİŞ SATIR ----
    assert "İpek Gömlek" in data1.get("clarification_options", []) 
    # ---- DÜZELTİLMİŞ SATIR ----
    assert "Hangisini sormuştunuz?" in data1.get("bot_response")

    # 2. Kullanıcı sadece ürün adını belirterek yanıt verir
    response2 = client.post("/process_query/", json={"query": "İpek Gömlek", "session_id": session_id})
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2.get("detected_intent") == "fiyat_sorgulama", \
        f"Beklenen niyet 'fiyat_sorgulama', alınan '{data2.get('detected_intent')}'"
    assert data2.get("resolved_product") == "İpek Gömlek", \
        f"Beklenen ürün 'İpek Gömlek', alınan '{data2.get('resolved_product')}'"
    assert "1250 TL" in data2.get("bot_response", ""), \
        f"Bot yanıtı '1250 TL' içermeli, alınan: '{data2.get('bot_response')}'"
    assert data2.get("ask_for_clarification") is False

@pytest.mark.asyncio
async def test_contextual_product_info_then_stok(client_fixture):
    client = client_fixture
    check_resources_loaded(client)
    session_id = "test_context_stok_1"

    # 1. Kullanıcı sadece ürün adını söyler
    response1 = client.post("/process_query/", json={"query": "Deri Ceket", "session_id": session_id})
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1.get("detected_intent") == "ürün_bilgisi_sorma"
    assert data1.get("resolved_product") == "Deri Ceket"
    assert "Hakiki kuzu derisinden" in data1.get("bot_response") # Ürün bilgisi verildi

    # 2. Kullanıcı aynı ürün için stok sorar
    response2 = client.post("/process_query/", json={"query": "stokta var mı bunun L bedeni", "session_id": session_id})
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2.get("detected_intent") == "stok_sorgulama"
    assert data2.get("resolved_product") == "Deri Ceket" # Bağlamdan ürün alındı
    assert data2.get("resolved_size") == "L"
    assert "'Deri Ceket' L bedeni için" in data2.get("bot_response")
    assert data2.get("ask_for_clarification") is False

@pytest.mark.asyncio
async def test_short_product_name_no_context_then_question(client_fixture):
    client = client_fixture
    check_resources_loaded(client)
    session_id = "test_short_prod_no_context"

    # Kullanıcı sadece ürün adını söyler (bağlam yok)
    response = client.post("/process_query/", json={"query": "Kot Pantolon", "session_id": session_id})
    assert response.status_code == 200
    data = response.json()
    assert data.get("detected_intent") == "ürün_bilgisi_sorma"
    assert data.get("resolved_product") == "Kot Pantolon"
    # Beklenti: Bot doğrudan ürün bilgisini verir, soru sormaz.
    assert "dayanıklı kot pantolon" in data.get("bot_response") 
    assert data.get("ask_for_clarification") is False

@pytest.mark.asyncio
async def test_product_name_after_failed_intent(client_fixture):
    client = client_fixture
    check_resources_loaded(client)
    session_id = "test_prod_after_fail"

    # 1. Kullanıcı anlamsız bir şey sorar
    response1 = client.post("/process_query/", json={"query": "asdfghjkl", "session_id": session_id})
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1.get("detected_intent") == "kapsam_disi"
    assert "Üzgünüm, bu konuda yardımcı olamıyorum" in data1.get("bot_response")

    # 2. Kullanıcı sadece ürün adını söyler
    response2 = client.post("/process_query/", json={"query": "İpek Gömlek", "session_id": session_id})
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2.get("detected_intent") == "ürün_bilgisi_sorma"
    assert data2.get("resolved_product") == "İpek Gömlek"
    assert "%100 saf ipekten" in data2.get("bot_response")
    assert data2.get("ask_for_clarification") is False
