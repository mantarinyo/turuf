#!/usr/bin/env python3
"""
Hızlı Test Modu - Ağır modeller olmadan test
"""
import requests
import time
import json

def test_fast_mode():
    """Fast mode'da chatbot testi"""
    
    # Test verileri
    test_cases = [
        {
            "query": "Bu pantolonun S bedeni var mı?",
            "expected_intent": "stok_sorgulama",
            "description": "Stok sorgulama"
        },
        {
            "query": "Bu elbisenin fiyatı ne kadar?",
            "expected_intent": "fiyat_sorgulama", 
            "description": "Fiyat sorgulama"
        },
        {
            "query": "Kargo ücreti ne kadar?",
            "expected_intent": "kargo_bilgisi_sorma",
            "description": "Kargo bilgisi"
        },
        {
            "query": "Merhaba",
            "expected_intent": "selamlama",
            "description": "Selamlama"
        },
        {
            "query": "İade var mı?",
            "expected_intent": "iade_sorgulama",
            "description": "İade sorgulama"
        }
    ]
    
    print("🚀 Hızlı Test Modu Başlıyor...")
    print("=" * 50)
    
    success_count = 0
    total_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"   Soru: {test_case['query']}")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                "http://localhost:8080/process_query/",
                json={
                    "query": test_case["query"],
                    "tenant_id": 1
                },
                timeout=10
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            total_time += response_time
            
            if response.status_code == 200:
                data = response.json()
                detected_intent = data.get("detected_intent", "unknown")
                
                if detected_intent == test_case["expected_intent"]:
                    print(f"   ✅ BAŞARILI - Intent: {detected_intent}")
                    print(f"   ⏱️  Süre: {response_time:.2f}s")
                    success_count += 1
                else:
                    print(f"   ❌ BAŞARISIZ - Beklenen: {test_case['expected_intent']}, Gelen: {detected_intent}")
                    print(f"   ⏱️  Süre: {response_time:.2f}s")
            else:
                print(f"   ❌ HTTP HATASI: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ TIMEOUT - 10 saniye aşıldı")
        except requests.exceptions.ConnectionError:
            print(f"   🔌 BAĞLANTI HATASI - Server çalışmıyor")
        except Exception as e:
            print(f"   💥 HATA: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST SONUÇLARI:")
    print(f"   Başarılı: {success_count}/{len(test_cases)}")
    print(f"   Başarı Oranı: {(success_count/len(test_cases)*100):.1f}%")
    print(f"   Ortalama Süre: {(total_time/len(test_cases)):.2f}s")
    
    if success_count == len(test_cases):
        print("🎉 TÜM TESTLER BAŞARILI!")
    else:
        print("⚠️  BAZI TESTLER BAŞARISIZ!")

if __name__ == "__main__":
    test_fast_mode() 