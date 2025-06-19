#!/usr/bin/env python3
# quick_test.py - Hızlı chatbot testi

import requests
import json
import time

# Test sunucusu URL'i
BASE_URL = "http://localhost:8000"

def test_chatbot():
    """Chatbot'u test et"""
    
    test_cases = [
        # Temel testler
        ("Merhaba", "selamlama"),
        ("Keten pantolon fiyatı", "fiyat_sorgulama"),
        ("S beden var mı?", "stok_sorgulama"),
        ("Bu ürünün malzemesi ne?", "ürün_malzeme_sorma"),
        
        # Yazım hataları
        ("slm", "selamlama"),
        ("fiyat ne kdr", "fiyat_sorgulama"),
        ("stokda varmı", "stok_sorgulama"),
        
        # Karmaşık sorular
        ("Bu pantolonun M bedeni var mı?", "stok_sorgulama"),
        ("İpek gömlek hakkında bilgi", "ürün_bilgisi_sorma"),
    ]
    
    print("=== CHATBOT TEST BAŞLIYOR ===\n")
    
    for i, (query, expected_intent) in enumerate(test_cases, 1):
        print(f"Test {i}: '{query}'")
        
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
                
                print(f"  ✅ Status: {response.status_code}")
                print(f"  🎯 Intent: {detected_intent} (beklenen: {expected_intent})")
                print(f"  💬 Cevap: {bot_response[:100]}...")
                
                if detected_intent == expected_intent:
                    print(f"  ✅ DOĞRU!")
                else:
                    print(f"  ❌ YANLIŞ! Beklenen: {expected_intent}")
                    
            else:
                print(f"  ❌ Hata: {response.status_code}")
                print(f"  📄 {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"  ❌ Bağlantı hatası - Sunucu çalışmıyor")
            print(f"  💡 Sunucuyu başlatmak için: uvicorn main:app --reload")
            break
        except Exception as e:
            print(f"  ❌ Hata: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_chatbot() 