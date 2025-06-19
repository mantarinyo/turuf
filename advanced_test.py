#!/usr/bin/env python3
# advanced_test.py - Karmaşık senaryolar test scripti

import requests
import json
import time

# Test sunucusu URL'i
BASE_URL = "http://127.0.0.1:8080"

def test_advanced_scenarios():
    """Karmaşık senaryoları test et"""
    
    test_cases = [
        # 1. KARMAŞIK ENTITY EXTRACTION
        ("Keten pantolonun M bedeni var mı ve fiyatı ne kadar?", "stok_sorgulama"),
        ("İpek gömlek hakkında bilgi ver, malzemesi ne?", "ürün_malzeme_sorma"),
        ("Deri ceket stokta mı, kaç para?", "stok_sorgulama"),
        
        # 2. ÇOKLU SORULAR
        ("Bu ürünün fiyatı ne kadar ve stokta var mı?", "fiyat_sorgulama"),
        ("Kargo ücreti ne kadar ve kaç günde gelir?", "kargo_bilgisi_sorma"),
        ("İade var mı ve nasıl yapılır?", "iade_sorgulama"),
        
        # 3. ZAMİRLİ KARMAŞIK SORULAR
        ("Bunun fiyatı ne kadar ve hangi bedenlerde var?", "fiyat_sorgulama"),
        ("Şunun malzemesi ne ve stokta mevcut mu?", "ürün_malzeme_sorma"),
        ("Onun özellikleri neler ve kaç para?", "ürün_bilgisi_sorma"),
        
        # 4. YAZIM HATALARI VE KISALTMALAR
        ("fiyat ne kdr bu ürünün", "fiyat_sorgulama"),
        ("stokda varmı M beden", "stok_sorgulama"),
        ("kargo nrd ve ne kdr", "kargo_bilgisi_sorma"),
        ("iade var mı ve nasıl", "iade_sorgulama"),
        
        # 5. İŞLETME BİLGİLERİ
        ("Mağazanız nerede ve çalışma saatleri nedir?", "lokasyon_sorma"),
        ("Telefon numaranız nedir ve nasıl ulaşabilirim?", "tel_no_sorma"),
        ("Hangi ödeme yöntemlerini kabul ediyorsunuz?", "odeme_yontemleri_sorma"),
        
        # 6. ÖNERİ VE TAVSİYE
        ("Ne önerirsin bu sezon için?", "oneri_isteme"),
        ("En çok satan ürününüz hangisi?", "oneri_isteme"),
        ("Bana bir şey öner", "oneri_isteme"),
        
        # 7. NEGATİF DURUMLAR
        ("Hayır, teşekkürler", "olumsuz_yanıt"),
        ("Yok kalsın", "olumsuz_yanıt"),
        ("İstemiyorum", "olumsuz_yanıt"),
        
        # 8. TEŞEKKÜR VE SELAMLAMA
        ("Teşekkürler", "tesekkur"),
        ("Sağ ol", "tesekkur"),
        ("Merhaba, nasılsınız?", "selamlama"),
        
        # 9. KARMAŞIK ÜRÜN SORULARI
        ("Bu elbisenin kumaşı ne ve hangi renklerde var?", "ürün_malzeme_sorma"),
        ("Pantolonun kalıbı dar mı ve hangi bedenlerde mevcut?", "ürün_bilgisi_sorma"),
        ("Ceketin astarı var mı ve nasıl yıkanır?", "ürün_bilgisi_sorma"),
        
        # 10. SİPARİŞ VE TESLİMAT
        ("Sipariş nasıl verilir ve ne kadar sürer?", "kargo_bilgisi_sorma"),
        ("Kapıda ödeme var mı ve taksit yapıyor musunuz?", "odeme_yontemleri_sorma"),
        ("Kargo takip edebilir miyim?", "kargo_bilgisi_sorma"),
    ]
    
    print("=== KARMAŞIK SENARYO TEST BAŞLIYOR ===\n")
    
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
                confidence = data.get("confidence", 0.0)
                clarification_needed = data.get("clarification_needed", False)
                
                print(f"  ✅ Status: {response.status_code}")
                print(f"  🎯 Intent: {detected_intent} (beklenen: {expected_intent})")
                print(f"  📊 Confidence: {confidence:.2f}")
                print(f"  📦 Entities: {entities}")
                print(f"  💬 Cevap: {bot_response[:80]}...")
                if clarification_needed:
                    print(f"  ❓ Netleştirme gerekli")
                
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
        
        print("-" * 70)
    
    # SONUÇ RAPORU
    print("\n" + "="*70)
    print("📊 KARMAŞIK SENARYO TEST SONUÇLARI")
    print("="*70)
    print(f"✅ Başarılı: {passed}/{total} (%{(passed/total)*100:.1f})")
    print(f"❌ Başarısız: {failed}/{total} (%{(failed/total)*100:.1f})")
    
    if passed == total:
        print("🎉 TÜM KARMAŞIK TESTLER BAŞARILI!")
    elif passed >= total * 0.8:
        print("👍 Çoğu karmaşık test başarılı, mükemmel performans!")
    elif passed >= total * 0.6:
        print("⚠️  Orta performans, bazı iyileştirmeler gerekli")
    else:
        print("🚨 Düşük performans, önemli iyileştirmeler gerekli")
    
    return passed, failed, total

if __name__ == "__main__":
    test_advanced_scenarios() 