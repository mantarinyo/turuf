#!/usr/bin/env python3
"""
Optimized Test Suite - Fast Mode Enabled
Bu test dosyası FAST_MODE kullanarak ağır modelleri devre dışı bırakır
ve testleri hızlandırır.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient
import time

# FAST_MODE'u aktif et
os.environ["FAST_MODE"] = "true"

# Projenin ana dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from main import app

@pytest.fixture(scope="session")
def client():
    """Session scope'da client oluştur - tüm testler için tek client"""
    with TestClient(app) as c:
        yield c

def test_app_startup(client):
    """Uygulama başlangıcını test et"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "FAST MODE AKTİF" in data["message"] or "Aktif" in data["message"]

def test_basic_query(client):
    """Temel sorgu testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "Merhaba", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 2.0  # 2 saniyeden az sürmeli
    data = response.json()
    assert data["bot_response"] is not None

def test_product_query(client):
    """Ürün sorgusu testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "Keten pantolonun fiyatı nedir?", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 2.0
    data = response.json()
    assert data["detected_intent"] is not None

def test_stock_query(client):
    """Stok sorgusu testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "Bu pantolonun S bedeni var mı?", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 2.0
    data = response.json()
    assert data["detected_intent"] in ["stok_sorgulama", "ürün_bilgisi_sorma"]

def test_price_query(client):
    """Fiyat sorgusu testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "Bu elbisenin fiyatı nedir?", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 2.0
    data = response.json()
    assert data["detected_intent"] == "fiyat_sorgulama"

def test_shipping_query(client):
    """Kargo sorgusu testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "Kargo ücreti ne kadar?", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 2.0
    data = response.json()
    assert data["detected_intent"] == "kargo_bilgisi_sorma"

def test_return_query(client):
    """İade sorgusu testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "İade etmek istiyorum nasıl olacak?", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 2.0
    data = response.json()
    assert data["detected_intent"] == "iade_sorgulama"

def test_payment_query(client):
    """Ödeme sorgusu testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "Hangi ödeme yöntemlerini kabul ediyorsunuz?", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 2.0
    data = response.json()
    assert data["detected_intent"] == "odeme_yontemleri_sorma"

def test_location_query(client):
    """Konum sorgusu testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "Mağazanız nerede?", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 2.0
    data = response.json()
    assert data["detected_intent"] == "lokasyon_sorma"

def test_contact_query(client):
    """İletişim sorgusu testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "Telefon numaranızı alabilir miyim?", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 2.0
    data = response.json()
    assert data["detected_intent"] == "tel_no_sorma"

def test_out_of_scope_query(client):
    """Kapsam dışı sorgu testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "bana bir şaka anlat", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 2.0
    data = response.json()
    assert data["detected_intent"] == "kapsam_disi"

def test_empty_query(client):
    """Boş sorgu testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 2.0

def test_short_query(client):
    """Kısa sorgu testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "Fiyat?", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 2.0
    data = response.json()
    assert data["ask_for_clarification"] == True

def test_multiple_questions(client):
    """Çoklu soru testi"""
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "Bu yeşil elbisenin M bedeni var mı ve kapıda ödeme yapabilir miyim kargo ne kadar sürer?", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 3.0  # Çoklu soru için biraz daha fazla süre
    data = response.json()
    assert data["bot_response"] is not None

def test_session_context(client):
    """Session bağlam testi"""
    session_id = "test_session_123"
    
    # İlk sorgu
    response1 = client.post(
        "/process_query/",
        json={"query": "İpek Gömlek", "tenant_id": 1, "session_id": session_id}
    )
    assert response1.status_code == 200
    
    # İkinci sorgu - bağlam kullanarak
    response2 = client.post(
        "/process_query/",
        json={"query": "fiyatı ne kadar?", "tenant_id": 1, "session_id": session_id}
    )
    assert response2.status_code == 200
    
    data2 = response2.json()
    assert data2["bot_response"] is not None

def test_performance_batch(client):
    """Toplu performans testi"""
    queries = [
        "Merhaba",
        "Keten pantolonun fiyatı nedir?",
        "Bu pantolonun S bedeni var mı?",
        "Kargo ücreti ne kadar?",
        "İade etmek istiyorum nasıl olacak?",
        "Hangi ödeme yöntemlerini kabul ediyorsunuz?",
        "Mağazanız nerede?",
        "Telefon numaranızı alabilir miyim?",
        "bana bir şaka anlat",
        "Fiyat?"
    ]
    
    start_time = time.time()
    
    for query in queries:
        response = client.post(
            "/process_query/",
            json={"query": query, "tenant_id": 1}
        )
        assert response.status_code == 200
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Toplam 10 sorgu {total_time:.2f} saniyede tamamlandı")
    assert total_time < 20.0  # 10 sorgu için 20 saniyeden az

if __name__ == "__main__":
    # Test çalıştırma
    pytest.main([__file__, "-v", "--tb=short"]) 