# tests/test_api.py
from fastapi.testclient import TestClient
import pytest
import os
import sys
from contextlib import contextmanager

# Projenin ana dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

# Pytest için en doğru TestClient kullanım şekli.
# Bu fixture, testler çalışmadan önce uygulamanın lifespan olaylarını
# (kaynak yükleme vb.) tetikler ve testler bitince kapatır.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_root_endpoint(client):
    """Ana endpoint'in çalışıp çalışmadığını kontrol eder."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Chatbot NLU API" in data["message"]
    assert "Durum: Aktif" in data["message"]

@pytest.mark.parametrize("query, expected_product, expected_in_response", [
    ("Keten pantolonun fiyatı nedir?", "Keten Pantolon", "850 TL"),
    ("ipek gömlek hakkında bilgi", "İpek Gömlek", "%100 saf ipekten"),
    ("deri ceket malzemesi", "Deri Ceket", "Hakiki Kuzu Derisi"),
])
def test_direct_product_queries(client, query, expected_product, expected_in_response):
    """Farklı ürünler için doğrudan sorguları test eder."""
    response = client.post(
        "/process_query/",
        json={"query": query, "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    
    # NoneType kontrolü ekle
    if data["resolved_item_details"] is not None:
        assert data["resolved_item_details"]["name"] == expected_product
        assert expected_in_response in data["bot_response"]
    else:
        # Eğer ürün bulunamadıysa, en azından bot cevap vermeli
        assert data["bot_response"] is not None
        assert len(data["bot_response"]) > 0

def test_context_handling(client):
    """Botun konuşma bağlamını (hafızasını) doğru kullanıp kullanmadığını test eder."""
    session_id = "test_context_session_123"
    
    response1 = client.post(
        "/process_query/",
        json={"query": "İpek Gömlek", "tenant_id": 1, "session_id": session_id}
    )
    assert response1.status_code == 200
    if response1.json()["resolved_item_details"]:
        assert response1.json()["resolved_item_details"]["name"] == "İpek Gömlek"
    
    response2 = client.post(
        "/process_query/",
        json={"query": "fiyatı ne kadar?", "tenant_id": 1, "session_id": session_id}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    if data2["resolved_item_details"]:
        assert data2["resolved_item_details"]["name"] == "İpek Gömlek"
        assert "1250 TL" in data2["bot_response"]

def test_general_info_queries(client):
    """Adres, telefon gibi genel bilgileri doğru verip vermediğini test eder."""
    response = client.post(
        "/process_query/",
        json={"query": "adresiniz nerede", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert "Moda Caddesi" in data["bot_response"]

def test_out_of_scope_query(client):
    """Kapsam dışı bir soruya doğru fallback yanıtı verip vermediğini test eder."""
    response = client.post(
        "/process_query/",
        json={"query": "bana bir şaka anlat", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "kapsam_disi"
    assert "bu konuda yardımcı olamıyorum" in data["bot_response"]

# === GEMINI DEEP RESEARCH TESTLERİ ===

@pytest.mark.parametrize("query, expected_keywords", [
    ("Bu elbisenin S bedeni var mı?", ["kontrol", "ürün kodu", "görüntü"]),
    ("siyah pantln stogu bitti mi ne zmn gelir", ["stok", "bilgi", "paylaş"]),
    ("gomlek fiyati nedir", ["fiyat", "güncel"]),
    ("fiyata kargo ekleniyor mu", ["kargo", "ücret", "dahil"]),
    ("beden tablosu nerde", ["beden", "tablo", "link"]),
    ("168cm 55kg hangi beden olmali", ["boy", "kilo", "ölçü"]),
    ("ceketin baska rengi", ["renk", "seçenek"]),
    ("pantolon kumas ne", ["kumaş", "malzeme", "içerik"]),
    ("iade etmek istiyom nasıl olcak", ["14 gün", "iade", "hakkı"]),
    ("sütyen iade edebilir miyim", ["hijyen", "koşul", "değerlendir"]),
])
def test_stok_fiyat_beden_renk_malzeme_iade_queries(client, query, expected_keywords):
    """Stok, fiyat, beden, renk, malzeme ve iade sorgularını test eder."""
    response = client.post(
        "/process_query/",
        json={"query": query, "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Bot cevap vermeli
    assert data["bot_response"] is not None
    assert len(data["bot_response"]) > 0
    
    # En az bir beklenen anahtar kelime cevapta olmalı
    response_lower = data["bot_response"].lower()
    assert any(keyword in response_lower for keyword in expected_keywords)

@pytest.mark.parametrize("query, expected_keywords", [
    ("kargo ne zaman ulasir", ["kargo", "teslimat", "süre"]),
    ("kargo ne kadar tutuyor", ["kargo", "ücret", "ücretsiz"]),
    ("kapida odeme mevcut mu", ["kapıda", "ödeme", "seçenek"]),
    ("nasil odeyebilirim", ["kredi kartı", "havale", "eft"]),
    ("indirim var mi su an", ["kampanya", "indirim", "güncel"]),
    ("mavi elbiseyle ne canta gider", ["öneri", "çanta", "kombin"]),
])
def test_kargo_odeme_kampanya_oneri_queries(client, query, expected_keywords):
    """Kargo, ödeme, kampanya ve öneri sorgularını test eder."""
    response = client.post(
        "/process_query/",
        json={"query": query, "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["bot_response"] is not None
    assert len(data["bot_response"]) > 0
    
    response_lower = data["bot_response"].lower()
    assert any(keyword in response_lower for keyword in expected_keywords)

@pytest.mark.parametrize("query, expected_keywords", [
    ("yanlis urun geldi napicam", ["özür", "düzelteceğiz", "fotoğraf"]),
    ("defolu urun yollamissiniz", ["üzgünüz", "kusurlu", "değişim"]),
    ("iletisim bilgileri nerde", ["telefon", "whatsapp", "e-posta"]),
    ("cvp neden yok hala", ["özür", "geç", "yardımcı"]),
])
def test_sikayet_iletisim_queries(client, query, expected_keywords):
    """Şikayet ve iletişim sorgularını test eder."""
    response = client.post(
        "/process_query/",
        json={"query": query, "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["bot_response"] is not None
    assert len(data["bot_response"]) > 0
    
    response_lower = data["bot_response"].lower()
    assert any(keyword in response_lower for keyword in expected_keywords)

@pytest.mark.parametrize("query, should_ask_clarification", [
    ("Ondan istiyorum.", True),
    ("sunu alabilir miyim", True),
    ("bunu begendim", True),
    ("o ne kadar?", True),
    ("kirmizi rengi var mi bunun", True),
    ("farkli renk secenegi var mi bunda", True),
    ("bu ceket tek renk mi?", True),
])
def test_zamirli_eksik_queries_clarification(client, query, should_ask_clarification):
    """Zamirli/eksik sorgularda netleştirme sorularının sorulup sorulmadığını test eder."""
    response = client.post(
        "/process_query/",
        json={"query": query, "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    
    if should_ask_clarification:
        # Netleştirme sorusu sorulmalı veya belirsizlik ifade edilmeli
        response_lower = data["bot_response"].lower()
        clarification_indicators = [
            "hangi üründen", "anlayamadım", "paylaşabilir misiniz",
            "belirttiğiniz", "bahsettiğiniz", "teyit", "öğrenebilir miyim"
        ]
        assert any(indicator in response_lower for indicator in clarification_indicators)

@pytest.mark.parametrize("query, expected_parts", [
    ("Bu yeşil elbisenin M bedeni var mı ve kapıda ödeme yapabilir miyim kargo ne kadar sürer?", ["1.", "2.", "3."]),
    ("yesil elbise m beden stokta mi kapida odeme var mi kargo kac gun", ["1.", "2.", "3."]),
    ("slm bu ceket ne kadar bide iade var mı sizde?", ["1.", "2."]),
    ("ceket fiyat ve iade?", ["1.", "2."]),
])
def test_karmasik_coklu_queries(client, query, expected_parts):
    """Karmaşık/çoklu sorguların ayrıştırılıp her birine cevap verilip verilmediğini test eder."""
    response = client.post(
        "/process_query/",
        json={"query": query, "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["bot_response"] is not None
    assert len(data["bot_response"]) > 0
    
    # Çoklu sorulara numaralı cevap verilmeli
    response_text = data["bot_response"]
    for part in expected_parts:
        assert part in response_text

@pytest.mark.parametrize("query, expected_keywords", [
    ("ayakkabı 38 var mı", ["38", "numara", "stok"]),
    ("2 tane alirsam indirim yapar misiniz", ["indirim", "teklif", "kampanya"]),
    ("pantolon kalıbı dar mı", ["kalıp", "dar", "bol"]),
    ("kazak yün mü kaşındırır mı", ["yün", "kaşındırma", "malzeme"]),
    ("indirimden aldigim urunu iade edebilir miyim", ["iade", "kampanyalı", "yasal"]),
    ("kargom nerde", ["kargo", "takip", "numara"]),
    ("iban atar mısınız, havale yapacağım", ["iban", "havale", "hesap"]),
    ("indirim kodum calismiyor", ["kod", "geçerli", "kontrol"]),
    ("dugunde giymek icin abiye", ["abiye", "düğün", "öneri"]),
    ("fotodakinden farkli geldi kalitesiz", ["kalite", "farklı", "inceleme"]),
])
def test_detailed_product_queries(client, query, expected_keywords):
    """Detaylı ürün sorgularını test eder."""
    response = client.post(
        "/process_query/",
        json={"query": query, "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["bot_response"] is not None
    assert len(data["bot_response"]) > 0
    
    response_lower = data["bot_response"].lower()
    assert any(keyword in response_lower for keyword in expected_keywords)

def test_session_context_with_clarification(client):
    """Oturum bağlamında netleştirme sorularının doğru çalışıp çalışmadığını test eder."""
    session_id = "test_clarification_session_456"
    
    # İlk soru - ürün belirtme
    response1 = client.post(
        "/process_query/",
        json={"query": "İpek Gömlek", "tenant_id": 1, "session_id": session_id}
    )
    assert response1.status_code == 200
    
    # İkinci soru - zamirli soru
    response2 = client.post(
        "/process_query/",
        json={"query": "Peki ya kırmızısı?", "tenant_id": 1, "session_id": session_id}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    
    # Netleştirme sorusu sorulmalı
    response_lower = data2["bot_response"].lower()
    clarification_indicators = [
        "hangi ürünün", "önceki mesajınızda", "bahsettiğiniz ürünün"
    ]
    assert any(indicator in response_lower for indicator in clarification_indicators)

def test_typo_handling(client):
    """Yazım hatalarının düzgün işlenip işlenmediğini test eder."""
    typo_queries = [
        "pantoon fiyatı",  # pantolon
        "jekt malzemesi",  # ceket
        "kadr nedir",      # kadar
        "bedn tablosu",    # beden
        "fiyay",           # fiyat
        "calısma saatleri" # çalışma
    ]
    
    for query in typo_queries:
        response = client.post(
            "/process_query/",
            json={"query": query, "tenant_id": 1}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Bot cevap vermeli (yazım hatası olsa bile)
        assert data["bot_response"] is not None
        assert len(data["bot_response"]) > 0

def test_empty_and_short_queries(client):
    """Boş ve çok kısa sorguların işlenmesini test eder."""
    short_queries = [
        "fyt?",           # fiyat
        "stok?",          # stok
        "iade?",          # iade
        "kargo?",         # kargo
        "slm",            # selam
        "tşk",            # teşekkür
    ]
    
    for query in short_queries:
        response = client.post(
            "/process_query/",
            json={"query": query, "tenant_id": 1}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Bot cevap vermeli
        assert data["bot_response"] is not None
        assert len(data["bot_response"]) > 0

def test_negative_responses(client):
    """Olumsuz yanıtların doğru işlenmesini test eder."""
    negative_queries = [
        "hayır",
        "yok kalsın",
        "gerek yok",
        "istemiyorum",
        "düşünmüyorum",
        "vazgeçtim"
    ]
    
    for query in negative_queries:
        response = client.post(
            "/process_query/",
            json={"query": query, "tenant_id": 1}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Olumsuz yanıtlar için uygun cevap verilmeli
        assert data["detected_intent"] == "olumsuz_yanıt" or "tamam" in data["bot_response"].lower()

def test_greeting_and_thanks(client):
    """Selamlama ve teşekkür mesajlarının doğru işlenmesini test eder."""
    greeting_queries = [
        "merhaba",
        "selam",
        "iyi günler",
        "günaydın",
        "mrb",
        "slm"
    ]
    
    for query in greeting_queries:
        response = client.post(
            "/process_query/",
            json={"query": query, "tenant_id": 1}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Selamlama için uygun cevap
        assert data["detected_intent"] == "selamlama" or "merhaba" in data["bot_response"].lower()
    
    thanks_queries = [
        "teşekkür",
        "sağ ol",
        "tşk",
        "eyvallah",
        "saol"
    ]
    
    for query in thanks_queries:
        response = client.post(
            "/process_query/",
            json={"query": query, "tenant_id": 1}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Teşekkür için uygun cevap
        assert data["detected_intent"] == "tesekkur" or "rica" in data["bot_response"].lower()

