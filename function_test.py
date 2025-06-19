#!/usr/bin/env python3
"""
Fonksiyon Test - TestClient kullanmadan doğrudan fonksiyonları test eder
"""

import os
import sys
import time
import asyncio

# FAST_MODE'u aktif et
os.environ["FAST_MODE"] = "true"

# Projenin ana dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def test_functions_directly():
    """Fonksiyonları doğrudan test et"""
    print("🚀 Fonksiyon Test Başlıyor...")
    print("=" * 50)
    
    try:
        # Import sadece gerekli fonksiyonlar
        print("📦 Fonksiyonlar import ediliyor...")
        from main import detect_intent_fast_mode, generate_response_fast_mode
        from main import get_sqlite_service
        print("✅ Fonksiyonlar import başarılı")
        
        # Test verileri
        test_queries = [
            ("Merhaba", "selamlama"),
            ("Keten pantolonun fiyatı nedir?", "fiyat_sorgulama"),
            ("Bu pantolonun S bedeni var mı?", "stok_sorgulama"),
            ("Bu ürünün malzemesi ne?", "ürün_malzeme_sorma"),
            ("Kargo ücreti ne kadar?", "kargo_bilgisi_sorma"),
            ("İade etmek istiyorum nasıl olacak?", "iade_sorgulama"),
        ]
        
        # Tenant settings mock
        tenant_settings = {
            "business_name": "Test Mağazası",
            "settings_json": {
                "default_responses": {
                    "greeting": "Merhaba! Size nasıl yardımcı olabilirim?",
                    "fallback": "Ne demek istediğinizi tam anlayamadım."
                }
            }
        }
        
        print(f"\n📋 {len(test_queries)} test sorgusu çalıştırılıyor...")
        
        success_count = 0
        
        for i, (query, expected_intent) in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: '{query}' ---")
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
                
                print(f"⏱️  Süre: {duration:.3f} saniye")
                print(f"🎯 Intent: {detected_intent} (beklenen: {expected_intent})")
                print(f"📊 Confidence: {confidence:.2f}")
                print(f"💬 Response: {response[:100]}...")
                print(f"❓ Clarification: {clarification_needed}")
                
                # Başarı kontrolü
                if detected_intent == expected_intent:
                    print("✅ DOĞRU!")
                    success_count += 1
                else:
                    print("❌ YANLIŞ!")
                    
            except Exception as e:
                print(f"💥 Hata: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 50)
        print(f"📊 Test Sonuçları: {success_count}/{len(test_queries)} başarılı")
        print(f"📈 Başarı Oranı: {success_count/len(test_queries)*100:.1f}%")
        print("=" * 50)
        
        if success_count == len(test_queries):
            print("🎉 Tüm testler başarılı!")
        else:
            print("⚠️  Bazı testler başarısız oldu.")
            
    except Exception as e:
        print(f"💥 Genel hata: {e}")
        import traceback
        traceback.print_exc()

def test_sqlite_service():
    """SQLite servisini test et"""
    print("\n🔧 SQLite Service Test...")
    print("=" * 30)
    
    try:
        from main import get_sqlite_service
        
        service = get_sqlite_service()
        print("✅ SQLite service oluşturuldu")
        
        # Tenant settings test
        tenant_settings = asyncio.run(service.get_tenant_settings(1))
        print(f"✅ Tenant settings: {tenant_settings is not None}")
        
        if tenant_settings:
            print(f"📄 Business name: {tenant_settings.get('business_name', 'N/A')}")
        
    except Exception as e:
        print(f"💥 SQLite test hatası: {e}")

if __name__ == "__main__":
    test_functions_directly()
    test_sqlite_service() 