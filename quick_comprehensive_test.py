#!/usr/bin/env python3
"""
Hızlı Kapsamlı Test - Ana senaryoları test eder
"""
import requests
import time
import json

def test_comprehensive_scenarios():
    """Ana senaryoları test eder"""
    
    base_url = "http://localhost:8080"
    
    # Test senaryoları
    test_cases = [
        # 1. ÜRÜN VE STOK
        {"query": "Bu pantolonun S bedeni var mı?", "expected": "stok_sorgulama", "desc": "Stok sorgulama"},
        {"query": "Bu elbisenin fiyatı ne kadar?", "expected": "fiyat_sorgulama", "desc": "Fiyat sorgulama"},
        {"query": "Bu pantolonun kumaşı nedir?", "expected": "ürün_malzeme_sorma", "desc": "Malzeme sorgulama"},
        
        # 2. KARGO VE TESLİMAT
        {"query": "Kargo ücreti ne kadar?", "expected": "kargo_bilgisi_sorma", "desc": "Kargo ücreti"},
        {"query": "Kargom ne zaman ulaşır?", "expected": "kargo_bilgisi_sorma", "desc": "Teslimat süresi"},
        {"query": "Hangi kargo firması ile çalışıyorsunuz?", "expected": "kargo_bilgisi_sorma", "desc": "Kargo firması"},
        
        # 3. İADE VE DEĞİŞİM
        {"query": "İade var mı?", "expected": "iade_sorgulama", "desc": "İade sorgulama"},
        {"query": "Değişim yapıyor musunuz?", "expected": "iade_sorgulama", "desc": "Değişim sorgulama"},
        {"query": "Kaç günde iade edebilirim?", "expected": "iade_sorgulama", "desc": "İade süresi"},
        
        # 4. ÖDEME VE SİPARİŞ
        {"query": "Kapıda ödeme var mı?", "expected": "odeme_yontemleri_sorma", "desc": "Kapıda ödeme"},
        {"query": "Hangi ödeme yöntemlerini kabul ediyorsunuz?", "expected": "odeme_yontemleri_sorma", "desc": "Ödeme yöntemleri"},
        {"query": "Nasıl sipariş verebilirim?", "expected": "odeme_yontemleri_sorma", "desc": "Sipariş süreci"},
        
        # 5. İLETİŞİM VE BİLGİ
        {"query": "Mağazanız var mı, adresiniz nedir?", "expected": "lokasyon_sorma", "desc": "Adres sorgulama"},
        {"query": "Telefon numaranızı alabilir miyim?", "expected": "tel_no_sorma", "desc": "Telefon sorgulama"},
        {"query": "Çalışma saatleriniz nedir?", "expected": "calisma_saatleri_sorma", "desc": "Çalışma saatleri"},
        
        # 6. SELAMLAMA VE GENEL
        {"query": "Merhaba", "expected": "selamlama", "desc": "Selamlama"},
        {"query": "Teşekkürler", "expected": "tesekkur", "desc": "Teşekkür"},
        {"query": "Hayır", "expected": "olumsuz_yanıt", "desc": "Olumsuz yanıt"},
        
        # 7. TYPO VE SOKAK DİLİ
        {"query": "slm kargo kac gun surer", "expected": "kargo_bilgisi_sorma", "desc": "Sokak dili kargo"},
        {"query": "mrb bu ürünün fiyatı ne kdr", "expected": "fiyat_sorgulama", "desc": "Sokak dili fiyat"},
        {"query": "baska renk varmi", "expected": "ürün_bilgisi_sorma", "desc": "Sokak dili renk"},
        
        # 8. ZAMİR KULLANIMI
        {"query": "Bunun fiyatı ne kadar?", "expected": "fiyat_sorgulama", "desc": "Zamir fiyat"},
        {"query": "Bunun S bedeni var mı?", "expected": "stok_sorgulama", "desc": "Zamir stok"},
        
        # 9. ÇOKLU SORU
        {"query": "36 beden var mı? Fiyatı ne kadar? Kargo ücreti ne kadar?", "expected": "stok_sorgulama", "desc": "Çoklu soru"},
        
        # 10. ÖZEL DURUMLAR
        {"query": "Müşteri temsilcisine bağlar mısınız?", "expected": "musteri_hizmetlerine_baglanma", "desc": "Müşteri temsilcisi"},
        {"query": "Siparişim eksik geldi.", "expected": "musteri_hizmetlerine_baglanma", "desc": "Şikayet"},
    ]
    
    print("🚀 Kapsamlı Test Başlıyor...")
    print("=" * 60)
    
    success_count = 0
    total_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{base_url}/process_query/",
                json={
                    "query": test_case["query"],
                    "tenant_id": 1
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                detected_intent = data.get("detected_intent", "bilinmiyor")
                confidence = data.get("confidence", 0)
                
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                if detected_intent == test_case["expected"]:
                    print(f"✅ Test {i:2d}: {test_case['desc']}")
                    print(f"   Soru: {test_case['query']}")
                    print(f"   Intent: {detected_intent} (Confidence: {confidence:.2f})")
                    print(f"   Süre: {elapsed_time:.2f}s")
                    success_count += 1
                else:
                    print(f"❌ Test {i:2d}: {test_case['desc']}")
                    print(f"   Soru: {test_case['query']}")
                    print(f"   Beklenen: {test_case['expected']}")
                    print(f"   Gelen: {detected_intent} (Confidence: {confidence:.2f})")
                    print(f"   Süre: {elapsed_time:.2f}s")
                
            else:
                print(f"❌ Test {i:2d}: HTTP Hatası - {response.status_code}")
                print(f"   Soru: {test_case['query']}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Test {i:2d}: Bağlantı Hatası - {e}")
            print(f"   Soru: {test_case['query']}")
        
        print()
    
    print("=" * 60)
    print(f"📊 TEST SONUÇLARI:")
    print(f"   Toplam Test: {len(test_cases)}")
    print(f"   Başarılı: {success_count}")
    print(f"   Başarı Oranı: {(success_count/len(test_cases)*100):.1f}%")
    print(f"   Toplam Süre: {total_time:.2f}s")
    print(f"   Ortalama Süre: {(total_time/len(test_cases)):.2f}s")
    
    if success_count == len(test_cases):
        print("🎉 TÜM TESTLER BAŞARILI!")
    elif success_count >= len(test_cases) * 0.8:
        print("✅ ÇOK İYİ PERFORMANS!")
    elif success_count >= len(test_cases) * 0.6:
        print("⚠️  ORTA PERFORMANS - İyileştirme gerekli")
    else:
        print("❌ DÜŞÜK PERFORMANS - Acil iyileştirme gerekli")

if __name__ == "__main__":
    test_comprehensive_scenarios() 