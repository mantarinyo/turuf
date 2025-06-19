#!/usr/bin/env python3
"""
Kapsamlı Fonksiyon Test - Tüm test senaryolarını kapsar
"""

import os
import sys
import time
import asyncio

# FAST_MODE'u aktif et
os.environ["FAST_MODE"] = "true"

# Projenin ana dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def run_comprehensive_tests():
    """Kapsamlı test çalıştır"""
    print("🚀 Kapsamlı Fonksiyon Test Başlıyor...")
    print("=" * 60)
    
    try:
        # Import sadece gerekli fonksiyonlar
        print("📦 Fonksiyonlar import ediliyor...")
        from main import detect_intent_fast_mode, generate_response_fast_mode
        print("✅ Fonksiyonlar import başarılı")
        
        # Kapsamlı test verileri
        comprehensive_tests = [
            # 1. ÜRÜN BİLGİSİ VE STOK DURUMU TESTLERİ
            ("Bu pantolonun S bedeni var mı?", "stok_sorgulama"),
            ("Bu pantolonun kumaşı nedir?", "ürün_malzeme_sorma"),
            ("Yeni gelen ürünleriniz var mı?", "ürün_bilgisi_sorma"),
            ("Bu bluzun başka renkleri var mı?", "ürün_bilgisi_sorma"),
            ("Bu ceketin iç astarı var mı?", "ürün_bilgisi_sorma"),
            
            # 2. FİYAT VE İNDİRİMLER TESTLERİ
            ("Bu elbisenin fiyatı nedir?", "fiyat_sorgulama"),
            ("Kargo ücreti ne kadar?", "kargo_bilgisi_sorma"),
            ("İndirim kodumu nasıl kullanabilirim?", "odeme_yontemleri_sorma"),
            ("Şu an devam eden bir kampanya var mı?", "odeme_yontemleri_sorma"),
            
            # 3. SİPARİŞ SÜREÇLERİ TESTLERİ
            ("Nasıl sipariş verebilirim?", "odeme_yontemleri_sorma"),
            ("Kapıda ödeme var mı?", "odeme_yontemleri_sorma"),
            ("Hangi ödeme yöntemlerini kabul ediyorsunuz?", "odeme_yontemleri_sorma"),
            ("Siparişimin durumunu nasıl öğrenebilirim?", "siparis_durumu_sorma"),
            
            # 4. KARGO VE TESLİMAT TESTLERİ
            ("Kargom ne zaman ulaşır?", "kargo_bilgisi_sorma"),
            ("Kargo takip numaramı nasıl alabilirim?", "kargo_bilgisi_sorma"),
            ("Hangi kargo firması ile çalışıyorsunuz?", "kargo_bilgisi_sorma"),
            
            # 5. İADE, DEĞİŞİM VE İPTAL TESTLERİ
            ("İade etmek istiyorum nasıl olacak?", "iade_sorgulama"),
            ("Değişim yapabilir miyim?", "iade_sorgulama"),
            ("İade süresi ne kadar?", "iade_sorgulama"),
            ("Defolu ürün geldi ne yapacağım?", "iade_sorgulama"),
            
            # 6. BEDEN VE ÖLÇÜ TESTLERİ
            ("Beden tablosu var mı?", "ürün_bilgisi_sorma"),
            ("168cm 55kg hangi beden olmalı?", "ürün_bilgisi_sorma"),
            ("Bu pantolonun kalıbı dar mı?", "ürün_bilgisi_sorma"),
            
            # 7. İLETİŞİM VE KONUM TESTLERİ
            ("Mağazanız nerede?", "lokasyon_sorma"),
            ("Telefon numaranızı alabilir miyim?", "tel_no_sorma"),
            ("Çalışma saatleriniz nedir?", "calisma_saatleri_sorma"),
            ("Web siteniz var mı?", "websitesi_sorma"),
            
            # 8. ÖNERİ VE TAVSİYE TESTLERİ
            ("Ne önerirsiniz?", "oneri_isteme"),
            ("En çok satan ürünleriniz neler?", "oneri_isteme"),
            ("Hediye için ne önerirsiniz?", "oneri_isteme"),
            
            # 9. MÜŞTERİ HİZMETLERİ TESTLERİ
            ("Müşteri temsilcisine bağlar mısınız?", "musteri_hizmetlerine_baglanma"),
            ("Bir yetkiliyle görüşmek istiyorum.", "musteri_hizmetlerine_baglanma"),
            
            # 10. SELAMLAMA VE TEŞEKKÜR TESTLERİ
            ("Merhaba", "selamlama"),
            ("Selam", "selamlama"),
            ("slm", "selamlama"),
            ("mrb", "selamlama"),
            ("Teşekkürler", "tesekkur"),
            ("Sağ ol", "tesekkur"),
            ("tşk", "tesekkur"),
            
            # 11. OLUMSUZ YANIT TESTLERİ
            ("Hayır kalsın", "olumsuz_yanıt"),
            ("Gerek yok", "olumsuz_yanıt"),
            ("İstemiyorum", "olumsuz_yanıt"),
            
            # 12. KAPSAM DIŞI TESTLERİ
            ("bana bir şaka anlat", "kapsam_disi"),
            ("hava nasıl", "kapsam_disi"),
            ("futbol maçı ne zaman", "kapsam_disi"),
        ]
        
        # Tenant settings mock
        tenant_settings = {
            "business_name": "Test Giyim Mağazası",
            "settings_json": {
                "default_responses": {
                    "greeting": "Merhaba! Size nasıl yardımcı olabilirim?",
                    "fallback": "Ne demek istediğinizi tam anlayamadım. Lütfen farklı bir şekilde sorabilir misiniz?",
                    "thanks": "Rica ederim! Başka bir konuda yardımcı olabilir miyim?",
                    "out_of_scope": "Üzgünüm, bu konuda yardımcı olamıyorum."
                }
            }
        }
        
        print(f"\n📋 {len(comprehensive_tests)} kapsamlı test sorgusu çalıştırılıyor...")
        
        success_count = 0
        total_time = 0
        results = []
        
        for i, (query, expected_intent) in enumerate(comprehensive_tests, 1):
            print(f"\n--- Test {i:2d}: '{query}' ---")
            start_time = time.time()
            
            try:
                # Intent detection
                detected_intent, confidence = detect_intent_fast_mode(query)
                
                # Response generation
                entities = {"item_name_candidate": "", "size": None}
                response, clarification_needed = generate_response_fast_mode(
                    detected_intent, entities, query, tenant_settings
                )
                
                end_time = time.time()
                duration = end_time - start_time
                total_time += duration
                
                print(f"⏱️  Süre: {duration:.3f} saniye")
                print(f"🎯 Intent: {detected_intent} (beklenen: {expected_intent})")
                print(f"📊 Confidence: {confidence:.2f}")
                print(f"💬 Response: {response[:80]}...")
                print(f"❓ Clarification: {clarification_needed}")
                
                # Başarı kontrolü
                is_success = detected_intent == expected_intent
                if is_success:
                    print("✅ DOĞRU!")
                    success_count += 1
                else:
                    print("❌ YANLIŞ!")
                
                results.append({
                    "test_id": i,
                    "query": query,
                    "expected": expected_intent,
                    "detected": detected_intent,
                    "confidence": confidence,
                    "duration": duration,
                    "success": is_success
                })
                    
            except Exception as e:
                print(f"💥 Hata: {e}")
                results.append({
                    "test_id": i,
                    "query": query,
                    "expected": expected_intent,
                    "detected": "ERROR",
                    "confidence": 0.0,
                    "duration": 0.0,
                    "success": False
                })
        
        # Sonuçları analiz et
        print("\n" + "=" * 60)
        print("📊 KAPSAMLI TEST SONUÇLARI")
        print("=" * 60)
        
        print(f"📈 Başarı Oranı: {success_count}/{len(comprehensive_tests)} ({success_count/len(comprehensive_tests)*100:.1f}%)")
        print(f"⏱️  Toplam Süre: {total_time:.3f} saniye")
        print(f"⚡ Ortalama Süre: {total_time/len(comprehensive_tests):.3f} saniye/test")
        
        # Intent bazında başarı oranları
        intent_stats = {}
        for result in results:
            intent = result["expected"]
            if intent not in intent_stats:
                intent_stats[intent] = {"total": 0, "success": 0}
            intent_stats[intent]["total"] += 1
            if result["success"]:
                intent_stats[intent]["success"] += 1
        
        print(f"\n🎯 Intent Bazında Başarı Oranları:")
        for intent, stats in intent_stats.items():
            success_rate = stats["success"] / stats["total"] * 100
            print(f"  {intent}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Başarısız testleri listele
        failed_tests = [r for r in results if not r["success"]]
        if failed_tests:
            print(f"\n❌ Başarısız Testler ({len(failed_tests)} adet):")
            for test in failed_tests:
                print(f"  Test {test['test_id']}: '{test['query']}' → {test['detected']} (beklenen: {test['expected']})")
        
        if success_count == len(comprehensive_tests):
            print("\n🎉 TÜM TESTLER BAŞARILI!")
        else:
            print(f"\n⚠️  {len(failed_tests)} test başarısız oldu.")
            
    except Exception as e:
        print(f"💥 Genel hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_tests() 