# tests/test_api_fixed.py
from fastapi.testclient import TestClient
import pytest
import os
import sys

# Test ortamı ayarları
os.environ["TESTING"] = "true"
os.environ["FAST_MODE"] = "true"

# Projenin ana dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test main'i import et
from test_main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_root_endpoint(client):
    """Ana endpoint'in çalışıp çalışmadığını kontrol eder."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Test Chatbot NLU API" in data["message"]
    assert "Test Modu Aktif" in data["message"]

def test_greeting_and_thanks(client):
    """Selamlama ve teşekkür testleri."""
    
    # Selamlama testi
    response1 = client.post(
        "/process_query/",
        json={"query": "Merhaba", "tenant_id": 1}
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["detected_intent"] == "selamlama"
    assert "Merhaba" in data1["bot_response"]
    
    # Teşekkür testi
    response2 = client.post(
        "/process_query/",
        json={"query": "Teşekkürler", "tenant_id": 1}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["detected_intent"] == "tesekkur"
    assert "Rica ederim" in data2["bot_response"]

def test_stock_and_price_queries(client):
    """Stok ve fiyat sorguları testleri."""
    
    # Stok sorgusu
    response1 = client.post(
        "/process_query/",
        json={"query": "Bu pantolonun S bedeni var mı?", "tenant_id": 1}
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["detected_intent"] == "stok_sorgulama"
    assert "stokta mevcut" in data1["bot_response"]
    
    # Fiyat sorgusu
    response2 = client.post(
        "/process_query/",
        json={"query": "Bu elbisenin fiyatı ne kadar?", "tenant_id": 1}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["detected_intent"] == "fiyat_sorgulama"
    assert "150 TL" in data2["bot_response"]

def test_shipping_and_return_queries(client):
    """Kargo ve iade sorguları testleri."""
    
    # Kargo sorgusu
    response1 = client.post(
        "/process_query/",
        json={"query": "Kargo ücreti ne kadar?", "tenant_id": 1}
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["detected_intent"] == "kargo_bilgisi_sorma"
    assert "2-3 günde" in data1["bot_response"]
    
    # İade sorgusu
    response2 = client.post(
        "/process_query/",
        json={"query": "İade var mı?", "tenant_id": 1}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["detected_intent"] == "iade_sorgulama"
    assert "14 gün" in data2["bot_response"]

def test_customer_service_queries(client):
    """Müşteri hizmetleri sorguları testleri."""
    
    # Müşteri temsilcisi
    response1 = client.post(
        "/process_query/",
        json={"query": "Müşteri temsilcisine bağlar mısınız?", "tenant_id": 1}
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["detected_intent"] == "musteri_hizmetlerine_baglanma"
    assert "müşteri temsilcisine" in data1["bot_response"]
    
    # Şikayet
    response2 = client.post(
        "/process_query/",
        json={"query": "Siparişim eksik geldi.", "tenant_id": 1}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["detected_intent"] == "musteri_hizmetlerine_baglanma"

def test_payment_and_location_queries(client):
    """Ödeme ve lokasyon sorguları testleri."""
    
    # Ödeme yöntemleri
    response1 = client.post(
        "/process_query/",
        json={"query": "Kapıda ödeme var mı?", "tenant_id": 1}
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["detected_intent"] == "odeme_yontemleri_sorma"
    assert "kapıda ödeme" in data1["bot_response"]
    
    # Lokasyon
    response2 = client.post(
        "/process_query/",
        json={"query": "Adresiniz nerede?", "tenant_id": 1}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["detected_intent"] == "lokasyon_sorma"
    assert "İstanbul" in data2["bot_response"]

def test_empty_query(client):
    """Boş sorgu testi."""
    response = client.post(
        "/process_query/",
        json={"query": "", "tenant_id": 1}
    )
    assert response.status_code == 400
    assert "boş olamaz" in response.json()["detail"]

def test_unknown_query(client):
    """Bilinmeyen sorgu testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu çok garip bir soru", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "bilinmiyor"
    assert "yardımcı olamıyorum" in data["bot_response"] 