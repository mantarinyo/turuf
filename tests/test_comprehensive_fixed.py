# tests/test_comprehensive_fixed.py
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

@pytest.fixture(scope="session")
def client():
    """Tüm testler için tek bir client kullan"""
    with TestClient(app) as c:
        yield c

# ============================================================================
# 1. TEMEL TESTLER
# ============================================================================

def test_root_endpoint(client):
    """Ana endpoint'in çalışıp çalışmadığını kontrol eder."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Test Chatbot NLU API" in data["message"]
    assert "Test Modu Aktif" in data["message"]

def test_greeting_and_thanks(client):
    """Selamlama ve teşekkür testleri."""
    # Selamlama
    response = client.post(
        "/process_query/",
        json={"query": "Merhaba", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "selamlama"
    assert data["confidence"] > 0.8
    
    # Teşekkür
    response = client.post(
        "/process_query/",
        json={"query": "Teşekkürler", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "tesekkur"
    assert data["confidence"] > 0.8

# ============================================================================
# 2. ÜRÜN BİLGİSİ VE STOK DURUMU TESTLERİ
# ============================================================================

def test_product_stock_query(client):
    """Ürün stok durumu sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu pantolonun S bedeni var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "stok_sorgulama"

def test_product_material_query(client):
    """Ürün malzeme bilgisi sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu pantolonun kumaşı nedir?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "urun_bilgisi_sorma"

def test_new_products_query(client):
    """Yeni ürünler sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Yeni gelen ürünleriniz var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "urun_bilgisi_sorma"

# ============================================================================
# 3. FİYAT VE İNDİRİMLER TESTLERİ
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

# ============================================================================
# 4. SİPARİŞ SÜREÇLERİ TESTLERİ
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

def test_payment_methods_query(client):
    """Ödeme yöntemleri sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Hangi ödeme yöntemlerini kabul ediyorsunuz?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "odeme_yontemleri_sorma"

# ============================================================================
# 5. KARGO VE TESLİMAT TESTLERİ
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

# ============================================================================
# 6. İADE, DEĞİŞİM VE İPTAL TESTLERİ
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
        json={"query": "Ürünü değiştirmek istiyorum, nasıl yapabilirim?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "iade_sorgulama"

# ============================================================================
# 7. İLETİŞİM BİLGİLERİ TESTLERİ
# ============================================================================

def test_contact_info_query(client):
    """İletişim bilgileri sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "İletişim bilgilerinizi alabilir miyim?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] in ["adres_sorma", "telefon_sorma"]

def test_location_query(client):
    """Konum sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Mağazanız nerede?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "adres_sorma"

def test_phone_query(client):
    """Telefon sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Telefon numaranız nedir?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "telefon_sorma"

# ============================================================================
# 8. YAZIM HATASI VE KISALTMALAR TESTLERİ
# ============================================================================

def test_typo_price_query(client):
    """Yazım hatası ile fiyat sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu ürünün fiyatı ne kdr?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "fiyat_sorgulama"

def test_abbreviation_stock_query(client):
    """Kısaltma ile stok sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Stokda varmı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "stok_sorgulama"

def test_slang_greeting(client):
    """Argo selamlama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "slm", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "selamlama"

# ============================================================================
# 9. ZAMİR KULLANIMI TESTLERİ
# ============================================================================

def test_pronoun_price_query(client):
    """Zamir ile fiyat sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bunun fiyatı ne kadar?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "fiyat_sorgulama"

def test_pronoun_stock_query(client):
    """Zamir ile stok sorgulama testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Bu ürünün stoku var mı?", "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "stok_sorgulama"

# ============================================================================
# 10. PERFORMANS TESTLERİ
# ============================================================================

def test_response_time_performance(client):
    """Yanıt süresi performans testi."""
    import time
    
    start_time = time.time()
    response = client.post(
        "/process_query/",
        json={"query": "Merhaba", "tenant_id": 1}
    )
    end_time = time.time()
    
    response_time = end_time - start_time
    
    assert response.status_code == 200
    assert response_time < 1.0  # 1 saniyeden az olmalı
    print(f"✅ Yanıt süresi: {response_time:.3f} saniye")

def test_concurrent_requests(client):
    """Eşzamanlı istek testi."""
    import threading
    import time
    
    results = []
    errors = []
    
    def make_request():
        try:
            response = client.post(
                "/process_query/",
                json={"query": "Merhaba", "tenant_id": 1}
            )
            results.append(response.status_code == 200)
        except Exception as e:
            errors.append(str(e))
    
    # 5 eşzamanlı istek
    threads = []
    for i in range(5):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    # Tüm thread'lerin bitmesini bekle
    for thread in threads:
        thread.join()
    
    assert len(errors) == 0, f"Eşzamanlı istek hataları: {errors}"
    assert all(results), "Tüm eşzamanlı istekler başarısız"

# ============================================================================
# 11. HATA DURUMLARI TESTLERİ
# ============================================================================

def test_empty_query(client):
    """Boş sorgu testi."""
    response = client.post(
        "/process_query/",
        json={"query": "", "tenant_id": 1}
    )
    assert response.status_code == 200  # Boş sorgu da işlenebilmeli

def test_missing_tenant_id(client):
    """Eksik tenant_id testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Merhaba"}
    )
    assert response.status_code == 422  # Validation error

def test_invalid_tenant_id(client):
    """Geçersiz tenant_id testi."""
    response = client.post(
        "/process_query/",
        json={"query": "Merhaba", "tenant_id": -1}
    )
    assert response.status_code == 200  # Geçersiz tenant_id de işlenebilmeli

def test_very_long_query(client):
    """Çok uzun sorgu testi."""
    long_query = "Merhaba " * 100  # 700 karakter
    response = client.post(
        "/process_query/",
        json={"query": long_query, "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == "selamlama"

# ============================================================================
# 12. KAPSAMLI MÜŞTERİ SORULARI TESTLERİ
# ============================================================================

@pytest.mark.parametrize("query,expected_intent", [
    ("Bu pantolonun M bedeni stokta var mı?", "stok_sorgulama"),
    ("36 beden kaldı mı bu elbiseden?", "stok_sorgulama"),
    ("Gömlek stokda varmı?", "stok_sorgulama"),
    ("Bu ceket hâlâ var mı stokta?", "stok_sorgulama"),
    ("Bu elbisenin fiyatı ne kadar?", "fiyat_sorgulama"),
    ("mrb bu ürünün fiyatı ne kdr", "fiyat_sorgulama"),
    ("Kaç para bu?", "fiyat_sorgulama"),
    ("L beden var mı?", "stok_sorgulama"),
    ("XL gelecek mi yine?", "stok_sorgulama"),
    ("Bedenler normal mi, kalıbı dar mı?", "stok_sorgulama"),
    ("Beden tablosu var mı?", "stok_sorgulama"),
    ("Bu modelin başka rengi var mı?", "urun_bilgisi_sorma"),
    ("Kırmızı rengi gelecek mi?", "stok_sorgulama"),
    ("Bu elbisenin siyahı var mı?", "stok_sorgulama"),
    ("Elbisenin kumaşı nedir?", "urun_bilgisi_sorma"),
    ("Bu pantolon pamuk mu?", "urun_bilgisi_sorma"),
    ("Markası ne bunun?", "urun_bilgisi_sorma"),
    ("Bu ürün orijinal mi?", "urun_bilgisi_sorma"),
    ("İade var mı?", "iade_sorgulama"),
    ("Kaç günde iade edebilirim?", "iade_sorgulama"),
    ("Değişim yapıyor musunuz?", "iade_sorgulama"),
    ("İndirimli ürünlerde iade oluyor mu?", "iade_sorgulama"),
    ("Mağazanız var mı, adresiniz nedir?", "adres_sorma"),
    ("Telefon numaranızı alabilir miyim?", "telefon_sorma"),
    ("Hangi ödeme yöntemleri mevcut?", "odeme_yontemleri_sorma"),
    ("Kapıda ödeme var mı?", "odeme_yontemleri_sorma"),
    ("Kredi kartına taksit yapıyor musunuz?", "odeme_yontemleri_sorma"),
    ("Kargo ücreti ne kadar?", "kargo_bilgisi_sorma"),
    ("Kargo kaç günde gelir?", "kargo_bilgisi_sorma"),
    ("Ücretsiz kargo var mı?", "kargo_bilgisi_sorma"),
    ("Yurt dışına gönderim var mı?", "kargo_bilgisi_sorma"),
    ("Hangi kargo ile çalışıyorsunuz?", "kargo_bilgisi_sorma"),
    ("Şu an bir kampanya var mı?", "odeme_yontemleri_sorma"),
    ("İndirim kodu var mı?", "odeme_yontemleri_sorma"),
    ("3 al 2 öde kampanyanız var mı?", "odeme_yontemleri_sorma"),
    ("İlk alışverişe indirim var mı?", "odeme_yontemleri_sorma"),
    ("Anneme hediye için ne önerirsiniz?", "urun_bilgisi_sorma"),
    ("Eşime pijama almak istiyorum, öneriniz var mı?", "urun_bilgisi_sorma"),
    ("Bu pantolonun yanına hangi gömlek gider?", "urun_bilgisi_sorma"),
    ("Müşteri temsilcisine bağlar mısınız?", "bilinmiyor"),
    ("Bir yetkiliyle görüşmek istiyorum.", "bilinmiyor"),
    ("Siparişim eksik geldi.", "bilinmiyor"),
    ("Aynı gün iki sipariş verdim, birleştirilsin.", "bilinmiyor"),
    ("Kargom hâlâ gelmedi.", "kargo_bilgisi_sorma"),
    ("Siparişimi iptal etmek istiyorum.", "iade_sorgulama"),
    ("Elbisenin bedeni olmadı. Değişim veya iade yapabilir miyim?", "iade_sorgulama"),
    ("Yanlış ürün gönderdiniz!", "bilinmiyor"),
    ("Ürün defolu çıktı, iade istiyorum.", "iade_sorgulama"),
    ("İç çamaşırı iade edilebilir mi?", "iade_sorgulama"),
    ("85C sütyen var mı?", "stok_sorgulama"),
    ("Boxer satıyor musunuz?", "stok_sorgulama"),
    ("Pijama var mı?", "stok_sorgulama"),
    ("Çorap var mı sizde?", "stok_sorgulama"),
    ("Instagram'da paylaştığınız kırmızı elbisenin linkini atar mısınız?", "urun_bilgisi_sorma"),
    ("36 beden var mı? Fiyatı ne kadar? Kargo ücreti ne kadar?", "stok_sorgulama"),
    ("slm kargo kac gun surer", "kargo_bilgisi_sorma"),
    ("mrb bu ürünün fiyatı ne kdr", "fiyat_sorgulama"),
    ("baska renk varmi", "stok_sorgulama"),
])
def test_comprehensive_customer_queries(client, query, expected_intent):
    """Kapsamlı müşteri soruları testi."""
    response = client.post(
        "/process_query/",
        json={"query": query, "tenant_id": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["detected_intent"] == expected_intent, f"Query: '{query}' - Expected: {expected_intent}, Got: {data['detected_intent']}"
    assert data["confidence"] > 0.1
    assert data["bot_response"] is not None
    assert len(data["bot_response"]) > 0 