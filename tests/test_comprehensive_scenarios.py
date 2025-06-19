# tests/test_comprehensive_scenarios.py
from fastapi.testclient import TestClient
import pytest
import os
import sys
import re

# Projenin ana dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# ============================================================================
# 1. ÜRÜN BİLGİSİ VE STOK DURUMU TESTLERİ
# ============================================================================

def test_product_stock_query(client):
    """Ürün stok durumu sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu pantolonun S bedeni var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    # Stok sorgusu için uygun intent tespit edilmeli
    assert data["detected_intent"] in ["stok_sorgulama", "ürün_bilgisi_sorma"]

def test_product_material_query(client):
    """Ürün malzeme bilgisi sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu pantolonun kumaşı nedir?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "ürün_malzeme_sorma"

def test_new_products_query(client):
    """Yeni ürünler sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Yeni gelen ürünleriniz var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "ürün_bilgisi_sorma"

def test_product_colors_query(client):
    """Ürün renk seçenekleri sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu bluzun başka renkleri var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "ürün_bilgisi_sorma"

def test_product_liner_query(client):
    """Ürün astar durumu sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu ceketin iç astarı var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "ürün_bilgisi_sorma"

# ============================================================================
# 2. FİYAT VE İNDİRİMLER TESTLERİ
# ============================================================================

def test_price_query(client):
    """Ürün fiyat sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu elbisenin fiyatı nedir?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "fiyat_sorgulama"

def test_shipping_cost_query(client):
    """Kargo ücreti sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Kargo ücreti ne kadar?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "kargo_bilgisi_sorma"

def test_discount_code_query(client):
    """İndirim kodu kullanımı sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "İndirim kodumu nasıl kullanabilirim?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "odeme_yontemleri_sorma"

def test_campaign_query(client):
    """Kampanya sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Şu an devam eden bir kampanya var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "odeme_yontemleri_sorma"

# ============================================================================
# 3. SİPARİŞ SÜREÇLERİ TESTLERİ
# ============================================================================

def test_how_to_order_query(client):
    """Nasıl sipariş verilir sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Nasıl sipariş verebilirim?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "odeme_yontemleri_sorma"

def test_cash_on_delivery_query(client):
    """Kapıda ödeme sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Kapıda ödeme var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "odeme_yontemleri_sorma"

def test_payment_methods_query(client):
    """Ödeme yöntemleri sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Hangi ödeme yöntemlerini kabul ediyorsunuz?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "odeme_yontemleri_sorma"

def test_order_status_query(client):
    """Sipariş durumu sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Siparişimin durumunu nasıl öğrenebilirim?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "siparis_durumu_sorma"

# ============================================================================
# 4. KARGO VE TESLİMAT TESTLERİ
# ============================================================================

def test_delivery_time_query(client):
    """Teslimat süresi sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Kargom ne zaman ulaşır?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "kargo_bilgisi_sorma"

def test_tracking_number_query(client):
    """Kargo takip numarası sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Kargo takip numaramı nasıl alabilirim?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "kargo_bilgisi_sorma"

def test_cargo_company_query(client):
    """Kargo firması sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Hangi kargo firması ile çalışıyorsunuz?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "kargo_bilgisi_sorma"

# ============================================================================
# 5. İADE, DEĞİŞİM VE İPTAL TESTLERİ
# ============================================================================

def test_return_process_query(client):
    """İade süreci sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Ürünü iade etmek istiyorum, süreci anlatır mısınız?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "iade_sorgulama"

def test_exchange_process_query(client):
    """Değişim süreci sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Değişim yapmak istiyorum, nasıl yapabilirim?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "iade_sorgulama"

def test_refund_time_query(client):
    """Para iadesi süresi sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Para iadem ne zaman yapılır?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "iade_sorgulama"

def test_defective_product_query(client):
    """Kusurlu ürün sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Kusurlu ürün geldi, ne yapmalıyım?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "iade_sorgulama"

# ============================================================================
# 6. BEDEN VE KALIP DESTEĞİ TESTLERİ
# ============================================================================

def test_size_chart_query(client):
    """Beden tablosu sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Beden tablonuz var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "ürün_bilgisi_sorma"

def test_fit_type_query(client):
    """Kalıp türü sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu elbisenin kalıbı dar mı geniş mi?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "ürün_bilgisi_sorma"

def test_model_measurements_query(client):
    """Manken ölçüleri sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Mankenin ölçüleri nedir?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "ürün_bilgisi_sorma"

# ============================================================================
# 7. MÜŞTERİ HİZMETLERİ VE ŞİKAYETLER TESTLERİ
# ============================================================================

def test_wrong_product_complaint(client):
    """Yanlış ürün şikayeti testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Yanlış ürün gönderdiniz.", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    # Şikayet için uygun intent tespit edilmeli
    assert data["detected_intent"] in ["iade_sorgulama", "musteri_hizmetlerine_baglanma"]

def test_contact_info_query(client):
    """İletişim bilgileri sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Size nasıl ulaşabilirim?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "tel_no_sorma"

def test_quality_complaint(client):
    """Kalite şikayeti testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Ürün kalitesinden memnun kalmadım.", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] in ["iade_sorgulama", "musteri_hizmetlerine_baglanma"]

# ============================================================================
# 8. YAZIM HATALARI VE KISALTMALAR TESTLERİ
# ============================================================================

def test_typo_price_query(client):
    """Yazım hatası ile fiyat sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu elbise ne kadar?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "fiyat_sorgulama"

def test_abbreviation_stock_query(client):
    """Kısaltma ile stok sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "S beden var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "stok_sorgulama"

def test_slang_greeting(client):
    """Argo ile selamlama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "slm", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "selamlama"

def test_phonetic_spelling(client):
    """Fonetik yazım testi."""
    response = client.post(
        "/process_query/",
        json={"query": "kargo nrd", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "kargo_bilgisi_sorma"

# ============================================================================
# 9. ZAMİRLİ SORULAR VE BAĞLAM TESTLERİ
# ============================================================================

def test_pronoun_price_query(client):
    """Zamirli fiyat sorgulama testi."""
    session_id = "test_pronoun_session_1"
    
    # İlk olarak bir ürün hakkında soru sor
    response1 = client.post(
        "/process_query/",
        json={"query": "Keten Pantolon hakkında bilgi", "tenant_id": 1, "session_id": session_id}
    )
    assert response1.status_code == 200
    
    # Sonra zamirli soru
    response2 = client.post(
        "/process_query/",
        json={"query": "Bu ürünün fiyatı nedir?", "tenant_id": 1, "session_id": session_id}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    # Bağlamdan anlamalı veya netleştirme sormalı
    assert data2["detected_intent"] == "fiyat_sorgulama"

def test_pronoun_stock_query(client):
    """Zamirli stok sorgulama testi."""
    session_id = "test_pronoun_session_2"
    
    # İlk olarak bir ürün hakkında soru sor
    response1 = client.post(
        "/process_query/",
        json={"query": "İpek Gömlek", "tenant_id": 1, "session_id": session_id}
    )
    assert response1.status_code == 200
    
    # Sonra zamirli soru
    response2 = client.post(
        "/process_query/",
        json={"query": "Bu ürünün stok durumu nedir?", "tenant_id": 1, "session_id": session_id}
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["detected_intent"] == "stok_sorgulama"

# ============================================================================
# 10. ÇOKLU ÜRÜN VE KARMAŞIK SORULAR TESTLERİ
# ============================================================================

def test_multiple_product_query(client):
    """Çoklu ürün sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Kırmızı elbise ve siyah gömlek stokta mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    # Çoklu ürün için uygun intent tespit edilmeli
    assert data["detected_intent"] in ["stok_sorgulama", "ürün_bilgisi_sorma"]

def test_complex_query(client):
    """Karmaşık soru testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Kırmızı elbise stokta mı, fiyatı ne?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    # Karmaşık soru için uygun intent tespit edilmeli
    assert data["detected_intent"] in ["stok_sorgulama", "fiyat_sorgulama", "ürün_bilgisi_sorma"]

# ============================================================================
# 11. İÇ GİYİM ÖZEL TESTLERİ
# ============================================================================

def test_lingerie_support_query(client):
    """İç giyim destek sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu sütyenin destekli mi?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "ürün_bilgisi_sorma"

def test_lingerie_size_query(client):
    """İç giyim beden sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu sütyenin X kap/beden ölçüsü var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "stok_sorgulama"

# ============================================================================
# 12. BÜYÜK BEDEN ÖZEL TESTLERİ
# ============================================================================

def test_plus_size_fit_query(client):
    """Büyük beden kalıp sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu büyük beden tunik esnek mi?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "ürün_bilgisi_sorma"

def test_plus_size_max_query(client):
    """Büyük beden maksimum sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu modelin en büyük bedeni nedir?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "stok_sorgulama"

# ============================================================================
# 13. TESETTÜR GİYİM ÖZEL TESTLERİ
# ============================================================================

def test_modest_length_query(client):
    """Tesettür giyim boy sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Tesettür elbisenin boyu kaç cm?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "ürün_bilgisi_sorma"

def test_modest_fabric_query(client):
    """Tesettür giyim kumaş sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu tuniğin kumaşı iç gösteriyor mu?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "ürün_bilgisi_sorma"

# ============================================================================
# 14. GENEL BİLGİLER TESTLERİ
# ============================================================================

def test_location_query(client):
    """Adres sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Adresiniz nerede?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "lokasyon_sorma"

def test_phone_query(client):
    """Telefon sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Telefon numaranız nedir?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "tel_no_sorma"

def test_working_hours_query(client):
    """Çalışma saatleri sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Çalışma saatleriniz nedir?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "calisma_saatleri_sorma"

def test_website_query(client):
    """Web sitesi sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Web siteniz var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "websitesi_sorma"

# ============================================================================
# 15. ÖNERİ VE TAVSİYE TESTLERİ
# ============================================================================

def test_recommendation_query(client):
    """Öneri isteme testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Ne önerirsin?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "oneri_isteme"

def test_bestseller_query(client):
    """En çok satan sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "En çok satan ürününüz hangisi?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "oneri_isteme"

# ============================================================================
# 16. CANLI DESTEK VE YETKİLİ BAĞLANTI TESTLERİ
# ============================================================================

def test_live_support_query(client):
    """Canlı destek isteme testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Yetkiliyle görüşmek istiyorum", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "musteri_hizmetlerine_baglanma"

def test_human_representative_query(client):
    """İnsan temsilci isteme testi."""
    response = client.post(
        "/process_query/",
        json={"query": "İnsanla konuşmak istiyorum", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "musteri_hizmetlerine_baglanma"

# ============================================================================
# 17. NEGATİF VE OLUMSUZ DURUMLAR TESTLERİ
# ============================================================================

def test_negative_response(client):
    """Olumsuz yanıt testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Hayır, teşekkürler", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "olumsuz_yanıt"

def test_out_of_scope_query(client):
    """Kapsam dışı soru testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bana bir şaka anlat", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "kapsam_disi"

# ============================================================================
# 18. SELAMLAMA VE TEŞEKKÜR TESTLERİ
# ============================================================================

def test_greeting_query(client):
    """Selamlama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Merhaba", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "selamlama"

def test_thanks_query(client):
    """Teşekkür testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Teşekkürler", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "tesekkur"

# ============================================================================
# 19. PERFORMANS VE HIZ TESTLERİ
# ============================================================================

def test_response_time_performance(client):
    """Yanıt süresi performans testi."""
    import time
    start_time = time.time()
    
    response = client.post(
        "/process_query/",
        json={"query": "Keten pantolon fiyatı", "tenant_id": 1}
    )
    
    end_time = time.time()
    response_time = end_time - start_time
    
    assert response.status_code == 200
    # Yanıt süresi 5 saniyeden az olmalı
    assert response_time < 5.0

def test_concurrent_requests(client):
    """Eşzamanlı istek testi."""
    import threading
    import time
    
    results = []
    
    def make_request():
        response = client.post(
            "/process_query/",
            json={"query": "Test sorgusu", "tenant_id": 1}
        )
        results.append(response.status_code)
    
    # 5 eşzamanlı istek
    threads = []
    for i in range(5):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    # Tüm thread'lerin bitmesini bekle
    for thread in threads:
        thread.join()
    
    # Tüm istekler başarılı olmalı
    assert all(status == 200 for status in results)
    assert len(results) == 5

# ============================================================================
# 20. HATA DURUMLARI VE EXCEPTION TESTLERİ
# ============================================================================

def test_empty_query(client):
    """Boş sorgu testi."""
    response = client.post(
        "/process_query/",
        json={"query": "", "tenant_id": 1}
    )
    assert response.status_code == 400

def test_missing_tenant_id(client):
    """Eksik tenant_id testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Test sorgusu"}
    )
    assert response.status_code == 422  # Validation error

def test_invalid_tenant_id(client):
    """Geçersiz tenant_id testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Test sorgusu", "tenant_id": 99999}
    )
    assert response.status_code == 404

def test_very_long_query(client):
    """Çok uzun sorgu testi."""
    long_query = "Bu çok uzun bir test sorgusu " * 100
    response = client.post(
        "/process_query/",
        json={"query": long_query, "tenant_id": 1}
    )
    assert response.status_code == 200  # Sistem uzun sorguları da işleyebilmeli 

# ============================================================================
# 21. KULLANICININ GÖNDERDİĞİ DEEP RESEARCH SORU-CEVAP TESTLERİ
# ============================================================================

@pytest.mark.parametrize("query,expected_response,clarification_needed", [
    ("Bu pantolonun M bedeni stokta var mı?", "Evet, M beden stokta mevcut.", False),
    ("36 beden kaldı mı bu elbiseden?", "Maalesef, 36 beden tükendi.", False),
    ("Gömlek stokda varmı?", "Evet, stokta var.", False),
    ("Bu ceket hâlâ var mı stokta?", "Evet, ürün stokta var.", False),
    ("Bu elbisenin fiyatı ne kadar?", "Fiyatı 350 TL.", False),
    ("Fiyat?", None, True),
    ("mrb bu ürünün fiyatı ne kdr", "Merhaba, ürünün fiyatı 150 TL.", False),
    ("Kaç para bu?", "Bu ürün 100 TL.", False),
    ("L beden var mı?", "Evet, L beden stokta var.", False),
    ("XL gelecek mi yine?", "Şu an XL beden yok, önümüzdeki hafta stokta olacak.", False),
    ("Bedenler normal mi, kalıbı dar mı?", "Ürün kalıbı normal, kendi bedeninizi tercih edebilirsiniz.", False),
    ("Beden tablosu var mı?", "Evet, beden tablosu sitemizde mevcuttur.", False),
    ("Bu modelin başka rengi var mı?", "Evet, bu modelin kırmızı ve mavi renkleri mevcut.", False),
    ("Kırmızı rengi gelecek mi?", "Evet, kırmızı rengi yakında stoklarımıza ekleyeceğiz.", False),
    ("Bu elbisenin siyahı var mı?", "Maalesef, bu modelin siyah rengi yok.", False),
    ("Elbisenin kumaşı nedir?", "Kumaşı %100 pamuktur.", False),
    ("Bu pantolon pamuk mu?", "Pantolonun kumaşı pamuk ve elastan karışımı.", False),
    ("Markası ne bunun?", "Butiğimizin kendi tasarımıdır.", False),
    ("Bu ürün orijinal mi?", "Evet, tüm ürünlerimiz orijinaldir.", False),
    ("İade var mı?", "Evet, teslimattan sonra 14 gün iade süremiz var.", False),
    ("Kaç günde iade edebilirim?", "Teslim aldıktan sonra 14 gün içinde iade edebilirsiniz.", False),
    ("Değişim yapıyor musunuz?", "Evet, 14 gün içinde değişim yapabilirsiniz.", False),
    ("İndirimli ürünlerde iade oluyor mu?", "Evet, indirimli ürünleri de 14 gün içinde iade edebilirsiniz.", False),
    ("Mağazanız var mı, adresiniz nedir?", "Evet, mağazamız var. Adresimiz: İstanbul, Kadıköy, Bahariye Caddesi No: 10.", False),
    ("Telefon numaranızı alabilir miyim?", "Müşteri hizmetleri numaramız: 0 (212) 345 67 89.", False),
    ("Hangi ödeme yöntemleri mevcut?", "Kredi kartı, banka havalesi ve kapıda ödeme seçeneklerimiz var.", False),
    ("Kapıda ödeme var mı?", "Evet, kapıda nakit veya kartla ödeme yapabilirsiniz.", False),
    ("Kredi kartına taksit yapıyor musunuz?", "Evet, kredi kartına 3 taksit imkanı var.", False),
    ("Kargo ücreti ne kadar?", "Kargo ücreti sabit 20 TL'dir.", False),
    ("Kargo kaç günde gelir?", "Kargoya verildikten sonra 2-3 gün içinde teslim edilir.", False),
    ("Ücretsiz kargo var mı?", "Evet, 300 TL ve üzeri siparişlerde kargo ücretsiz.", False),
    ("Yurt dışına gönderim var mı?", "Evet, yurt dışına da gönderim yapıyoruz.", False),
    ("Hangi kargo ile çalışıyorsunuz?", "Yurtiçi Kargo ile çalışıyoruz.", False),
    ("Şu an bir kampanya var mı?", "Evet, şu an tüm ürünlerde %10 indirim kampanyamız var.", False),
    ("İndirim kodu var mı?", "Maalesef, şu an geçerli bir indirim kodumuz yok.", False),
    ("3 al 2 öde kampanyanız var mı?", "Maalesef, 3 al 2 öde kampanyamız bulunmuyor.", False),
    ("İlk alışverişe indirim var mı?", "Evet, ilk alışverişinizde %5 indirim uyguluyoruz.", False),
    ("Anneme hediye için ne önerirsiniz?", "Elbise veya şık bir şal hediye için uygun olabilir.", False),
    ("Eşime pijama almak istiyorum, öneriniz var mı?", "Pamuklu pijama takımlarımız hediye için ideal olacaktır.", False),
    ("Bu pantolonun yanına hangi gömlek gider?", "Beyaz veya siyah bir gömlek bu pantolonla uyumlu olur.", False),
    ("Müşteri temsilcisine bağlar mısınız?", "Tabii, hemen bağlıyorum.", False),
    ("Bir yetkiliyle görüşmek istiyorum.", "Elbette, sizi hemen yetkiliye aktarıyorum.", False),
    ("Siparişim eksik geldi.", "Yaşanan eksiklik için özür dileriz, hemen yetkiliye aktarıyorum.", False),
    ("Aynı gün iki sipariş verdim, birleştirilsin.", "Elbette, iki siparişinizi tek pakette birleştiriyoruz.", False),
    ("Kargom hâlâ gelmedi.", "Gecikme için üzgünüz. Takip kodunuzu paylaşırsanız hemen kontrol edelim.", False),
    ("Siparişimi iptal etmek istiyorum.", "Tamam, iptal talebinizi alıyorum ve işleme koyuyorum.", False),
    ("Elbisenin bedeni olmadı. Değişim veya iade yapabilir miyim?", "Evet, ürünü 14 gün içinde iade edebilir veya farklı bedenle değiştirebilirsiniz.", False),
    ("Yanlış ürün gönderdiniz!", "Yaşanan karışıklık için özür dileriz, doğru ürünü göndermek üzere hemen işlem yapıyoruz.", False),
    ("Ürün defolu çıktı, iade istiyorum.", "Özür dileriz. Defolu ürünü ücretsiz iade alabiliriz, hemen yardımcı oluyoruz.", False),
    ("İç çamaşırı iade edilebilir mi?", "Maalesef, hijyen nedeniyle iç çamaşırı ürünlerinde iade veya değişim yapamıyoruz.", False),
    ("85C sütyen var mı?", "Evet, 85C beden sütyen stoklarımızda mevcut.", False),
    ("Boxer satıyor musunuz?", "Evet, erkek boxer modellerimiz mevcuttur.", False),
    ("Pijama var mı?", "Evet, çeşitli pijama modellerimiz var.", False),
    ("Çorap var mı sizde?", "Evet, farklı renk ve desende çoraplarımız bulunuyor.", False),
    ("Instagram'da paylaştığınız kırmızı elbisenin linkini atar mısınız?", "Tabii, ilgili ürünün satış linkini paylaşıyorum: [URL]", False),
    ("36 beden var mı? Fiyatı ne kadar? Kargo ücreti ne kadar?", "36 beden stokta mevcut. Fiyatı 250 TL. Kargo ücreti 20 TL.", False),
    ("slm kargo kac gun surer", "Merhaba, kargo genellikle 2-3 günde teslim edilir.", False),
    ("mrb bu ürünün fiyatı ne kdr", "Merhaba, ürünün fiyatı 180 TL.", False),
    ("baska renk varmi", "Bu ürünün başka rengi bulunmuyor.", False),
    ("Fiyat??", None, True),
])
def test_deepresearch_customer_qa(client, query, expected_response, clarification_needed):
    """
    Kullanıcının deep research tablosundaki müşteri soruları için chatbotun doğru cevabı döndürüp döndürmediğini test eder.
    Netleştirme gereken sorularda, chatbotun uygun netleştirme sorusu sorduğu kontrol edilir.
    """
    response = client.post(
        "/process_query/",
        json={"query": query, "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    answer = data.get("answer") or data.get("response") or data.get("message")
    if clarification_needed:
        # Netleştirme beklenen sorularda, chatbotun netleştirme sorusu sorduğu kontrol edilir
        assert re.search(r'(hangi|lütfen|ürün adı|ürünü belirt|ürünü paylaş|ürün bilgisi|ürün linki|ürün kodu)', answer.lower()), f"Netleştirme bekleniyordu, cevap: {answer}"
    else:
        # Cevap tam eşleşmeli veya çok yakın olmalı
        assert expected_response.lower().strip('.') in answer.lower(), f"Beklenen: {expected_response}, Gelen: {answer}" 