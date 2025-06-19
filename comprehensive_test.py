#!/usr/bin/env python3
# comprehensive_test.py - Kapsamlı test scripti

import requests
import json
import time

# Test sunucusu URL'i
BASE_URL = "http://127.0.0.1:8080"

def test_all_scenarios():
    """Tüm test senaryolarını çalıştır"""
    
    test_cases = [
        # 1. TEMEL TESTLER
        ("Merhaba", "selamlama"),
        ("slm", "selamlama"),
        ("mrb", "selamlama"),
        
        # 2. FİYAT SORGULAMA
        ("Keten pantolon fiyatı", "fiyat_sorgulama"),
        ("fiyat ne kdr", "fiyat_sorgulama"),
        ("Bu ürünün fiyatı nedir?", "fiyat_sorgulama"),
        ("kaç para", "fiyat_sorgulama"),
        
        # 3. STOK SORGULAMA
        ("S beden var mı?", "stok_sorgulama"),
        ("stokda varmı", "stok_sorgulama"),
        ("M bedeni mevcut mu?", "stok_sorgulama"),
        ("42 numara var mı?", "stok_sorgulama"),
        
        # 4. MALZEME SORGULAMA
        ("Bu ürünün malzemesi ne?", "ürün_malzeme_sorma"),
        ("kumaş nedir", "ürün_malzeme_sorma"),
        ("içerik ne", "ürün_malzeme_sorma"),
        
        # 5. KARMAŞIK SORULAR
        ("Bu pantolonun M bedeni var mı?", "stok_sorgulama"),
        ("İpek gömlek hakkında bilgi", "bilinmiyor"),
        ("Deri ceket fiyatı ne kadar?", "fiyat_sorgulama"),
        
        # 6. YAZIM HATALARI
        ("fiyat ne kdr", "fiyat_sorgulama"),
        ("stokda varmı", "stok_sorgulama"),
        ("kargo nrd", "bilinmiyor"),
        
        # 7. ZAMİRLİ SORULAR
        ("Bunun fiyatı ne kadar?", "fiyat_sorgulama"),
        ("Bu ürünün stoku var mı?", "stok_sorgulama"),
        ("Şunun malzemesi ne?", "ürün_malzeme_sorma"),
    ]
    
    print("=== KAPSAMLI CHATBOT TEST BAŞLIYOR ===\n")
    
    passed = 0
    failed = 0
    total = len(test_cases)
    
    for i, (query, expected_intent) in enumerate(test_cases, 1):
        print(f"Test {i:2d}/{total}: '{query}'")
        
        try:
            response = requests.post(
                f"{BASE_URL}/process_query/",
                json={"query": query, "tenant_id": 1},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                detected_intent = data.get("detected_intent", "bilinmiyor")
                bot_response = data.get("bot_response", "cevap yok")
                entities = data.get("entities", {})
                
                print(f"  ✅ Status: {response.status_code}")
                print(f"  🎯 Intent: {detected_intent} (beklenen: {expected_intent})")
                print(f"  📦 Entities: {entities}")
                print(f"  💬 Cevap: {bot_response[:80]}...")
                
                if detected_intent == expected_intent:
                    print(f"  ✅ DOĞRU!")
                    passed += 1
                else:
                    print(f"  ❌ YANLIŞ! Beklenen: {expected_intent}")
                    failed += 1
                    
            else:
                print(f"  ❌ Hata: {response.status_code}")
                print(f"  📄 {response.text}")
                failed += 1
                
        except requests.exceptions.ConnectionError:
            print(f"  ❌ Bağlantı hatası")
            failed += 1
        except Exception as e:
            print(f"  ❌ Hata: {e}")
            failed += 1
        
        print("-" * 60)
    
    # SONUÇ RAPORU
    print("\n" + "="*60)
    print("📊 TEST SONUÇLARI")
    print("="*60)
    print(f"✅ Başarılı: {passed}/{total} (%{(passed/total)*100:.1f})")
    print(f"❌ Başarısız: {failed}/{total} (%{(failed/total)*100:.1f})")
    
    if passed == total:
        print("🎉 TÜM TESTLER BAŞARILI!")
    elif passed >= total * 0.8:
        print("👍 Çoğu test başarılı, iyi performans!")
    elif passed >= total * 0.6:
        print("⚠️  Orta performans, iyileştirme gerekli")
    else:
        print("🚨 Düşük performans, önemli sorunlar var")
    
    return passed, failed, total

if __name__ == "__main__":
    test_all_scenarios() 