#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time
import random
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:8000"
TENANT_ID = 1

def test_api_endpoint(query, expected_intent=None, description=""):
    """API endpoint'ini test et"""
    try:
        payload = {
            "query": query,
            "tenant_id": TENANT_ID,
            "session_id": f"test_session_{random.randint(1000, 9999)}"
        }
        
        response = requests.post(f"{BASE_URL}/process_query/", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            detected_intent = data.get("detected_intent", "")
            bot_response = data.get("bot_response", "")
            
            print(f"\n🔍 TEST: {description}")
            print(f"📝 Query: '{query}'")
            print(f"🎯 Detected Intent: {detected_intent}")
            print(f"🤖 Response: {bot_response[:200]}{'...' if len(bot_response) > 200 else ''}")
            
            if expected_intent and detected_intent != expected_intent:
                print(f"⚠️  BEKLENEN: {expected_intent}, BULUNAN: {detected_intent}")
            else:
                print(f"✅ Intent doğru!")
                
            return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

def test_product_search():
    """Ürün arama testleri"""
    print("\n" + "="*50)
    print("🛍️  ÜRÜN ARAMA TESTLERİ")
    print("="*50)
    
    # Temel ürün arama testleri
    test_cases = [
        ("pijama takımları göster", "ürün_arama", "Temel pijama arama"),
        ("gecelik var mı", "ürün_arama", "Gecelik arama"),
        ("dantelli sabahlık", "ürün_arama", "Özellik ile arama"),
        ("siyah elbise", "ürün_arama", "Renk + ürün arama"),
        ("hamile pijama", "ürün_arama", "Özel kategori arama"),
        ("büyük beden gecelik", "ürün_arama", "Beden + ürün arama"),
    ]
    
    for query, expected_intent, description in test_cases:
        test_api_endpoint(query, expected_intent, description)
        time.sleep(1)

def test_price_queries():
    """Fiyat sorgulama testleri"""
    print("\n" + "="*50)
    print("💰 FİYAT SORGULAMA TESTLERİ")
    print("="*50)
    
    test_cases = [
        ("dantelli gecelik fiyat", "fiyat_sorgulama", "Ürün + fiyat"),
        ("pijama takımı ne kadar", "fiyat_sorgulama", "Ne kadar sorusu"),
        ("sabahlık kaç para", "fiyat_sorgulama", "Kaç para sorusu"),
        ("hamile pijama fiyatı nedir", "fiyat_sorgulama", "Uzun fiyat sorusu"),
        ("fiyatlar ne kadar", "fiyat_sorgulama", "Genel fiyat sorusu"),
    ]
    
    for query, expected_intent, description in test_cases:
        test_api_endpoint(query, expected_intent, description)
        time.sleep(1)

def test_spelling_correction():
    """Yazım hatası düzeltme testleri"""
    print("\n" + "="*50)
    print("✏️  YAZIM HATASI DÜZELTMe TESTLERİ")
    print("="*50)
    
    test_cases = [
        ("pijma takımı", "ürün_arama", "Pijama yazım hatası"),
        ("geclik var mı", "ürün_arama", "Gecelik yazım hatası"),
        ("sabahlik göster", "ürün_arama", "Sabahlık yazım hatası"),
        ("danteli elbise", "ürün_arama", "Dantelli yazım hatası"),
        ("siyah pjama", "ürün_arama", "Siyah pijama yazım hatası"),
        ("kirmizi gecelik", "ürün_arama", "Kırmızı yazım hatası"),
    ]
    
    for query, expected_intent, description in test_cases:
        test_api_endpoint(query, expected_intent, description)
        time.sleep(1)

def test_intent_detection():
    """Intent detection testleri"""
    print("\n" + "="*50)
    print("🎯 INTENT DETECTION TESTLERİ")
    print("="*50)
    
    test_cases = [
        ("merhaba", "selamlama", "Temel selamlama"),
        ("kategoriler neler", "kategori_listesi", "Kategori listesi"),
        ("stokta var mı", "stok_sorgulama", "Stok sorgulama"),
        ("iade yapabiliyor musunuz", "iade_sorgulama", "İade sorgulama"),
        ("kargo ne kadar", "kargo_bilgisi_sorma", "Kargo bilgisi"),
        ("teşekkürler", "tesekkur", "Teşekkür"),
        ("çalışma saatleri", "calisma_saatleri_sorma", "Çalışma saatleri"),
        ("adresiniz nedir", "lokasyon_sorma", "Lokasyon sorgulama"),
        ("telefon numaranız", "tel_no_sorma", "Telefon sorgulama"),
        ("nasıl ödeme yapabilirim", "odeme_yontemleri_sorma", "Ödeme yöntemleri"),
    ]
    
    for query, expected_intent, description in test_cases:
        test_api_endpoint(query, expected_intent, description)
        time.sleep(1)

def test_session_context():
    """Session context testleri"""
    print("\n" + "="*50)
    print("💭 SESSION CONTEXT TESTLERİ")
    print("="*50)
    
    session_id = f"context_test_{random.randint(1000, 9999)}"
    
    # Aynı session'da ardışık sorular
    queries = [
        ("pijama takımları göster", "İlk ürün arama"),
        ("bunların fiyatları ne kadar", "Context ile fiyat sorgulama"),
        ("başka renkleri var mı", "Context ile renk sorgulama"),
        ("teşekkürler", "Teşekkür")
    ]
    
    for query, description in queries:
        try:
            payload = {
                "query": query,
                "tenant_id": TENANT_ID,
                "session_id": session_id
            }
            
            response = requests.post(f"{BASE_URL}/process_query/", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n🔍 {description}")
                print(f"📝 Query: '{query}'")
                print(f"🎯 Intent: {data.get('detected_intent', '')}")
                print(f"🤖 Response: {data.get('bot_response', '')[:150]}...")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Session test error: {e}")

def test_edge_cases():
    """Edge case testleri"""
    print("\n" + "="*50)
    print("🚨 EDGE CASE TESTLERİ")
    print("="*50)
    
    test_cases = [
        ("", "bilinmiyor", "Boş query"),
        ("a", "bilinmiyor", "Tek karakter"),
        ("asdfghjkl", "bilinmiyor", "Anlamsız string"),
        ("pijama pijama pijama pijama", "ürün_arama", "Tekrarlı kelime"),
        ("PİJAMA TAKIMI GÖSTER", "ürün_arama", "Büyük harf"),
        ("pijama??? takımı???", "ürün_arama", "Çoklu soru işareti"),
        ("pijama fiyat stok var mı kargo", "fiyat_sorgulama", "Çoklu intent"),
        ("123456", "bilinmiyor", "Sadece rakam"),
        ("!@#$%^&*()", "bilinmiyor", "Sadece özel karakter"),
    ]
    
    for query, expected_intent, description in test_cases:
        test_api_endpoint(query, expected_intent, description)
        time.sleep(1)

def test_performance():
    """Performance testleri"""
    print("\n" + "="*50)
    print("⚡ PERFORMANCE TESTLERİ")
    print("="*50)
    
    queries = [
        "pijama takımları göster",
        "gecelik fiyatları",
        "stokta var mı",
        "kargo ne kadar",
        "merhaba"
    ]
    
    total_time = 0
    success_count = 0
    
    for i in range(10):  # 10 test
        query = random.choice(queries)
        start_time = time.time()
        
        try:
            payload = {
                "query": query,
                "tenant_id": TENANT_ID,
                "session_id": f"perf_test_{i}"
            }
            
            response = requests.post(f"{BASE_URL}/process_query/", json=payload, timeout=30)
            
            if response.status_code == 200:
                success_count += 1
                
            elapsed = time.time() - start_time
            total_time += elapsed
            
            print(f"Test {i+1}: {elapsed:.3f}s - {query[:30]}...")
            
        except Exception as e:
            print(f"Performance test {i+1} failed: {e}")
        
        time.sleep(0.5)
    
    if success_count > 0:
        avg_time = total_time / success_count
        print(f"\n📊 Performance Özeti:")
        print(f"✅ Başarılı: {success_count}/10")
        print(f"⏱️  Ortalama süre: {avg_time:.3f}s")
        print(f"🚀 RPS: {1/avg_time:.1f}")

def test_product_api_endpoints():
    """Ürün API endpoint testleri"""
    print("\n" + "="*50)
    print("🔌 ÜRÜN API ENDPOINT TESTLERİ")
    print("="*50)
    
    # Products listesi
    try:
        response = requests.get(f"{BASE_URL}/products?limit=5", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ /products endpoint: {data['total']} ürün, {data['categories']} kategori")
        else:
            print(f"❌ /products endpoint error: {response.status_code}")
    except Exception as e:
        print(f"❌ /products endpoint error: {e}")
    
    # Product search
    try:
        response = requests.get(f"{BASE_URL}/products/search?q=pijama&limit=3", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ /products/search endpoint: {len(data['products'])} ürün bulundu")
        else:
            print(f"❌ /products/search endpoint error: {response.status_code}")
    except Exception as e:
        print(f"❌ /products/search endpoint error: {e}")

def main():
    """Ana test fonksiyonu"""
    print("🚀 KAPSAMLI TEST BAŞLATIYOR...")
    print(f"📍 Server: {BASE_URL}")
    print(f"🏢 Tenant ID: {TENANT_ID}")
    print(f"⏰ Test Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Server health check
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Server çalışıyor")
        else:
            print(f"⚠️  Server health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Server'a bağlanılamıyor: {e}")
        return
    
    # Test kategorileri
    test_product_search()
    test_price_queries()
    test_spelling_correction()
    test_intent_detection()
    test_session_context()
    test_edge_cases()
    test_performance()
    test_product_api_endpoints()
    
    print("\n" + "="*50)
    print("🎉 TÜM TESTLER TAMAMLANDI!")
    print("="*50)

if __name__ == "__main__":
    main() 