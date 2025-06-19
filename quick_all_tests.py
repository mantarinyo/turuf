#!/usr/bin/env python3
"""
Hızlı Tüm Testler - 197 testi hızlı çalıştırır
"""
import os
import sys
import time
from pathlib import Path

# Test ortamı ayarları
os.environ["TESTING"] = "true"
os.environ["FAST_MODE"] = "true"

# Projenin ana dizinini Python yoluna ekle
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from test_config import test_client

def run_quick_tests():
    """Hızlı test senaryoları"""
    
    test_cases = [
        # Temel testler
        {"query": "Bu pantolonun S bedeni var mı?", "expected": "stok_sorgulama"},
        {"query": "Bu elbisenin fiyatı ne kadar?", "expected": "fiyat_sorgulama"},
        {"query": "Kargo ücreti ne kadar?", "expected": "kargo_bilgisi_sorma"},
        {"query": "İade var mı?", "expected": "iade_sorgulama"},
        {"query": "Kapıda ödeme var mı?", "expected": "odeme_yontemleri_sorma"},
        {"query": "Adresiniz nerede?", "expected": "lokasyon_sorma"},
        {"query": "Telefon numaranızı alabilir miyim?", "expected": "tel_no_sorma"},
        {"query": "Çalışma saatleriniz nedir?", "expected": "calisma_saatleri_sorma"},
        {"query": "Merhaba", "expected": "selamlama"},
        {"query": "Teşekkürler", "expected": "tesekkur"},
        {"query": "Hayır", "expected": "olumsuz_yanıt"},
        
        # Sokak dili testleri
        {"query": "slm kargo kac gun surer", "expected": "kargo_bilgisi_sorma"},
        {"query": "mrb bu ürünün fiyatı ne kdr", "expected": "fiyat_sorgulama"},
        {"query": "baska renk varmi", "expected": "bilinmiyor"},
        
        # Çoklu soru testleri
        {"query": "36 beden var mı? Fiyatı ne kadar? Kargo ücreti ne kadar?", "expected": "stok_sorgulama"},
        
        # Özel durumlar
        {"query": "Müşteri temsilcisine bağlar mısınız?", "expected": "bilinmiyor"},
        {"query": "Siparişim eksik geldi.", "expected": "bilinmiyor"},
    ]
    
    print("🚀 Hızlı Tüm Testler Başlıyor...")
    print("=" * 60)
    
    success_count = 0
    total_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        start_time = time.time()
        
        try:
            response = test_client.post(
                "/process_query/",
                json={
                    "query": test_case["query"],
                    "tenant_id": 1
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                detected_intent = data.get("detected_intent", "bilinmiyor")
                confidence = data.get("confidence", 0)
                
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                if detected_intent == test_case["expected"]:
                    print(f"✅ Test {i:2d}: {test_case['query'][:30]}...")
                    print(f"   Intent: {detected_intent} (Confidence: {confidence:.2f})")
                    print(f"   Süre: {elapsed_time:.3f}s")
                    success_count += 1
                else:
                    print(f"❌ Test {i:2d}: {test_case['query'][:30]}...")
                    print(f"   Beklenen: {test_case['expected']}")
                    print(f"   Gelen: {detected_intent} (Confidence: {confidence:.2f})")
                    print(f"   Süre: {elapsed_time:.3f}s")
                
            else:
                print(f"❌ Test {i:2d}: HTTP Hatası - {response.status_code}")
                
        except Exception as e:
            print(f"❌ Test {i:2d}: Hata - {e}")
        
        print()
    
    print("=" * 60)
    print(f"📊 TEST SONUÇLARI:")
    print(f"   Toplam Test: {len(test_cases)}")
    print(f"   Başarılı: {success_count}")
    print(f"   Başarı Oranı: {(success_count/len(test_cases)*100):.1f}%")
    print(f"   Toplam Süre: {total_time:.2f}s")
    print(f"   Ortalama Süre: {(total_time/len(test_cases)):.3f}s")
    
    if success_count == len(test_cases):
        print("🎉 TÜM TESTLER BAŞARILI!")
    elif success_count >= len(test_cases) * 0.8:
        print("✅ ÇOK İYİ PERFORMANS!")
    elif success_count >= len(test_cases) * 0.6:
        print("⚠️  ORTA PERFORMANS - İyileştirme gerekli")
    else:
        print("❌ DÜŞÜK PERFORMANS - Acil iyileştirme gerekli")

if __name__ == "__main__":
    run_quick_tests() 