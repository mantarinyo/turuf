#!/usr/bin/env python3
"""
120 Testli Comprehensive Test - Fonksiyon testi olarak
"""

import os
import sys
import time
import re

# FAST_MODE'u aktif et
os.environ["FAST_MODE"] = "true"

# Projenin ana dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def run_120_comprehensive_tests():
    """120 testli comprehensive test çalıştır"""
    print("🚀 120 Testli Comprehensive Test Başlıyor...")
    print("=" * 70)
    
    try:
        # Import sadece gerekli fonksiyonlar
        print("📦 Fonksiyonlar import ediliyor...")
        from main import detect_intent_fast_mode, generate_response_fast_mode
        print("✅ Fonksiyonlar import başarılı")
        
        # 120 testli comprehensive test verileri
        comprehensive_120_tests = [
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
            
            # 13. DEEP RESEARCH CUSTOMER QA TESTLERİ (120 test)
            ("Bu pantolonun M bedeni stokta var mı?", "stok_sorgulama"),
            ("36 beden kaldı mı bu elbiseden?", "stok_sorgulama"),
            ("Gömlek stokda varmı?", "stok_sorgulama"),
            ("Bu ceket hâlâ var mı stokta?", "stok_sorgulama"),
            ("Bu elbisenin fiyatı ne kadar?", "fiyat_sorgulama"),
            ("Fiyat?", "fiyat_sorgulama"),
            ("mrb bu ürünün fiyatı ne kdr", "fiyat_sorgulama"),
            ("Kaç para bu?", "fiyat_sorgulama"),
            ("L beden var mı?", "stok_sorgulama"),
            ("XL gelecek mi yine?", "stok_sorgulama"),
            ("Bedenler normal mi, kalıbı dar mı?", "ürün_bilgisi_sorma"),
            ("Beden tablosu var mı?", "ürün_bilgisi_sorma"),
            ("Bu modelin başka rengi var mı?", "ürün_bilgisi_sorma"),
            ("Kırmızı rengi gelecek mi?", "stok_sorgulama"),
            ("Bu elbisenin siyahı var mı?", "stok_sorgulama"),
            ("Elbisenin kumaşı nedir?", "ürün_malzeme_sorma"),
            ("Bu pantolon pamuk mu?", "ürün_malzeme_sorma"),
            ("Markası ne bunun?", "ürün_bilgisi_sorma"),
            ("Bu ürün orijinal mi?", "ürün_bilgisi_sorma"),
            ("İade var mı?", "iade_sorgulama"),
            ("Kaç günde iade edebilirim?", "iade_sorgulama"),
            ("Değişim yapıyor musunuz?", "iade_sorgulama"),
            ("İndirimli ürünlerde iade oluyor mu?", "iade_sorgulama"),
            ("Mağazanız var mı, adresiniz nedir?", "lokasyon_sorma"),
            ("Telefon numaranızı alabilir miyim?", "tel_no_sorma"),
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
            ("Anneme hediye için ne önerirsiniz?", "oneri_isteme"),
            ("Eşime pijama almak istiyorum, öneriniz var mı?", "oneri_isteme"),
            ("Bu pantolonun yanına hangi gömlek gider?", "oneri_isteme"),
            ("Müşteri temsilcisine bağlar mısınız?", "musteri_hizmetlerine_baglanma"),
            ("Bir yetkiliyle görüşmek istiyorum.", "musteri_hizmetlerine_baglanma"),
            ("Siparişim eksik geldi.", "siparis_durumu_sorma"),
            ("Aynı gün iki sipariş verdim, birleştirilsin.", "siparis_durumu_sorma"),
            ("Kargom hâlâ gelmedi.", "siparis_durumu_sorma"),
            ("Siparişimi iptal etmek istiyorum.", "siparis_durumu_sorma"),
            ("Elbisenin bedeni olmadı. Değişim veya iade yapabilir miyim?", "iade_sorgulama"),
            ("Yanlış ürün gönderdiniz!", "siparis_durumu_sorma"),
            ("Ürün defolu çıktı, iade istiyorum.", "iade_sorgulama"),
            ("İç çamaşırı iade edilebilir mi?", "iade_sorgulama"),
            ("85C sütyen var mı?", "stok_sorgulama"),
            ("Boxer satıyor musunuz?", "stok_sorgulama"),
            ("Pijama var mı?", "stok_sorgulama"),
            ("Çorap var mı sizde?", "stok_sorgulama"),
            ("Instagram'da paylaştığınız kırmızı elbisenin linkini atar mısınız?", "websitesi_sorma"),
            ("36 beden var mı? Fiyatı ne kadar? Kargo ücreti ne kadar?", "stok_sorgulama"),
            ("slm kargo kac gun surer", "kargo_bilgisi_sorma"),
            ("mrb bu ürünün fiyatı ne kdr", "fiyat_sorgulama"),
            ("baska renk varmi", "ürün_bilgisi_sorma"),
            ("Fiyat??", "fiyat_sorgulama"),
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
        
        print(f"\n📋 {len(comprehensive_120_tests)} kapsamlı test sorgusu çalıştırılıyor...")
        
        success_count = 0
        total_time = 0
        results = []
        
        for i, (query, expected_intent) in enumerate(comprehensive_120_tests, 1):
            print(f"\n--- Test {i:3d}: '{query}' ---")
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
                print(f"💬 Response: {response[:60]}...")
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
        print("\n" + "=" * 70)
        print("📊 120 TESTLİ KAPSAMLI TEST SONUÇLARI")
        print("=" * 70)
        
        print(f"📈 Başarı Oranı: {success_count}/{len(comprehensive_120_tests)} ({success_count/len(comprehensive_120_tests)*100:.1f}%)")
        print(f"⏱️  Toplam Süre: {total_time:.3f} saniye")
        print(f"⚡ Ortalama Süre: {total_time/len(comprehensive_120_tests):.3f} saniye/test")
        
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
            for test in failed_tests[:20]:  # İlk 20'sini göster
                print(f"  Test {test['test_id']}: '{test['query']}' → {test['detected']} (beklenen: {test['expected']})")
            if len(failed_tests) > 20:
                print(f"  ... ve {len(failed_tests) - 20} test daha")
        
        if success_count == len(comprehensive_120_tests):
            print("\n🎉 TÜM 120 TEST BAŞARILI!")
        else:
            print(f"\n⚠️  {len(failed_tests)} test başarısız oldu.")
            
    except Exception as e:
        print(f"💥 Genel hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_120_comprehensive_tests() 