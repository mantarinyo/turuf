# tests/test_clarification.py
from fastapi.testclient import TestClient
import pytest
import os
import sys

# Projenin ana dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_ambiguous_product_query(client):
    """Belirsiz ürün sorgularında netleştirme yapılıp yapılmadığını test eder."""
    response = client.post(
        "/process_query/",
        json={"query": "pantolonun fiyatı nedir?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    
    print("clarification_options:", data.get("clarification_options"))
    print("resolved_item_details:", data.get("resolved_item_details"))
    
    # Netleştirme yapılması gerekiyor çünkü birden fazla pantolon var
    assert data["ask_for_clarification"] == True
    assert data["clarification_options"] is not None
    assert len(data["clarification_options"]) > 1
    
    # Netleştirme mesajında ürün isimleri olmalı
    response_text = data["bot_response"]
    assert "Hangisini sormuştunuz" in response_text or "birkaç seçenek buldum" in response_text

def test_clarification_response_handling(client):
    """Netleştirme sorusuna verilen cevabın doğru işlenip işlenmediğini test eder."""
    session_id = "test_clarification_session_456"
    
    # İlk sorgu - belirsiz
    response1 = client.post(
        "/process_query/",
        json={"query": "pantolonun fiyatı nedir?", "tenant_id": 1, "session_id": session_id}
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["ask_for_clarification"] == True
    
    # Netleştirme seçeneklerinden birini seç
    clarification_options = data1["clarification_options"]
    if clarification_options:
        selected_product_name = clarification_options[0]["name"]
        
        # Netleştirme cevabı
        response2 = client.post(
            "/process_query/",
            json={"query": selected_product_name, "tenant_id": 1, "session_id": session_id}
        )
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Artık netleştirme yapılmamalı ve doğru ürün seçilmeli
        assert data2["ask_for_clarification"] == False
        assert data2["resolved_item_details"]["name"] == selected_product_name
        assert "fiyat" in data2["bot_response"].lower()

def test_context_with_pronouns(client):
    """"Bu", "şu" gibi zamirlerle yapılan sorguların bağlamdan anlaşılmasını test eder."""
    session_id = "test_pronoun_session_789"
    
    # İlk olarak bir ürün hakkında soru sor
    response1 = client.post(
        "/process_query/",
        json={"query": "Keten Pantolon hakkında bilgi", "tenant_id": 1, "session_id": session_id}
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["resolved_item_details"]["name"] == "Keten Pantolon"
    
    # Sonra "bu ürünün fiyatı nedir?" diye sor
    response2 = client.post(
        "/process_query/",
        json={"query": "bu ürünün fiyatı nedir?", "tenant_id": 1, "session_id": session_id}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    
    # Bot bağlamdan "bu ürün"ün Keten Pantolon olduğunu anlamalı
    assert data2["resolved_item_details"]["name"] == "Keten Pantolon"
    assert "fiyat" in data2["bot_response"].lower()

def test_no_context_fallback(client):
    """Bağlam olmadan "bu ürün" gibi sorgulara nasıl cevap verdiğini test eder."""
    session_id = "test_no_context_session_999"
    
    # Yeni session ile direkt "bu ürünün fiyatı nedir?" sorusu
    response = client.post(
        "/process_query/",
        json={"query": "bu ürünün fiyatı nedir?", "tenant_id": 1, "session_id": session_id}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Bağlam olmadığı için netleştirme yapmalı veya açıklayıcı mesaj vermeli
    assert data["ask_for_clarification"] == True or "hangi ürün" in data["bot_response"].lower()

def test_multiple_similar_products(client):
    """Benzer isimli ürünler için netleştirme mantığını test eder."""
    response = client.post(
        "/process_query/",
        json={"query": "gömlek fiyatı", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Birden fazla gömlek varsa netleştirme yapmalı
    if data["ask_for_clarification"]:
        assert len(data["clarification_options"]) > 1
        assert "Hangisini sormuştunuz" in data["bot_response"] or "birkaç seçenek buldum" in data["bot_response"]

def test_specific_product_query_no_clarification(client):
    """Spesifik ürün sorgularında netleştirme yapılmamasını test eder."""
    response = client.post(
        "/process_query/",
        json={"query": "Keten Pantolon fiyatı nedir?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Spesifik ürün adı verildiğinde netleştirme yapılmamalı
    assert data["ask_for_clarification"] == False
    assert data["resolved_item_details"]["name"] == "Keten Pantolon"
    assert "fiyat" in data["bot_response"].lower()

def test_clarification_with_partial_answer(client):
    """Netleştirme sorusuna kısmi cevap verildiğinde doğru ürünü bulmalı."""
    session_id = "test_partial_clarification_1"
    # İlk sorgu - belirsiz
    response1 = client.post(
        "/process_query/",
        json={"query": "pantolon fiyat", "tenant_id": 1, "session_id": session_id}
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["ask_for_clarification"] == True
    clarification_options = data1["clarification_options"]
    # Kısmi cevap: "keten"
    response2 = client.post(
        "/process_query/",
        json={"query": "keten", "tenant_id": 1, "session_id": session_id}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["ask_for_clarification"] == False
    assert data2["resolved_item_details"] is not None
    assert "Keten Pantolon" in data2["resolved_item_details"]["name"]
    assert "fiyat" in data2["bot_response"].lower()

def test_clarification_with_wrong_answer(client):
    """Netleştirme sorusuna yanlış/uyumsuz cevap verildiğinde uygun hata mesajı dönmeli."""
    session_id = "test_wrong_clarification_1"
    # İlk sorgu - belirsiz
    response1 = client.post(
        "/process_query/",
        json={"query": "pantolon fiyat", "tenant_id": 1, "session_id": session_id}
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["ask_for_clarification"] == True
    # Yanlış cevap: "kırmızı"
    response2 = client.post(
        "/process_query/",
        json={"query": "kırmızı", "tenant_id": 1, "session_id": session_id}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    # Netleştirme devam etmeli veya uygun hata mesajı dönmeli
    assert data2["resolved_item_details"] is None
    assert data2["ask_for_clarification"] == True or "hangi ürün" in data2["bot_response"].lower() or "bulunmuyor" in data2["bot_response"].lower()

def test_clarification_with_exact_answer(client):
    """Netleştirme sorusuna tam ürün adıyla cevap verildiğinde doğru ürünü bulmalı."""
    session_id = "test_exact_clarification_1"
    # İlk sorgu - belirsiz
    response1 = client.post(
        "/process_query/",
        json={"query": "pantolon fiyat", "tenant_id": 1, "session_id": session_id}
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["ask_for_clarification"] == True
    clarification_options = data1["clarification_options"]
    # Tam cevap: "Kot Pantolon"
    response2 = client.post(
        "/process_query/",
        json={"query": "Kot Pantolon", "tenant_id": 1, "session_id": session_id}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["ask_for_clarification"] == False
    assert data2["resolved_item_details"] is not None
    assert "Kot Pantolon" in data2["resolved_item_details"]["name"]
    assert "fiyat" in data2["bot_response"].lower() 