#!/usr/bin/env python3
"""
Doğrudan Test - Pytest kullanmadan
"""

import os
import sys
import time
import asyncio
from fastapi.testclient import TestClient

# FAST_MODE'u aktif et
os.environ["FAST_MODE"] = "true"

# Projenin ana dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def run_direct_tests():
    """Doğrudan test çalıştır"""
    print("🚀 Doğrudan Test Başlıyor...")
    print("=" * 50)
    
    try:
        # Import main app
        print("📦 Main app import ediliyor...")
        from main import app
        print("✅ Main app import başarılı")
        
        # TestClient oluştur
        print("🔧 TestClient oluşturuluyor...")
        client = TestClient(app)
        print("✅ TestClient oluşturuldu")
        
        # Test 1: Ana endpoint
        print("\n📋 Test 1: Ana endpoint")
        start_time = time.time()
        response = client.get("/")
        end_time = time.time()
        
        print(f"⏱️  Süre: {end_time - start_time:.2f} saniye")
        print(f"📊 Status: {response.status_code}")
        print(f"📄 Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Test 1: Başarılı")
        else:
            print("❌ Test 1: Başarısız")
        
        # Test 2: Basit sorgu
        print("\n📋 Test 2: Basit sorgu")
        start_time = time.time()
        response = client.post(
            "/process_query/",
            json={"query": "Merhaba", "tenant_id": 1}
        )
        end_time = time.time()
        
        print(f"⏱️  Süre: {end_time - start_time:.2f} saniye")
        print(f"📊 Status: {response.status_code}")
        print(f"📄 Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Test 2: Başarılı")
        else:
            print("❌ Test 2: Başarısız")
        
        # Test 3: Ürün sorgusu
        print("\n📋 Test 3: Ürün sorgusu")
        start_time = time.time()
        response = client.post(
            "/process_query/",
            json={"query": "Keten pantolonun fiyatı nedir?", "tenant_id": 1}
        )
        end_time = time.time()
        
        print(f"⏱️  Süre: {end_time - start_time:.2f} saniye")
        print(f"📊 Status: {response.status_code}")
        print(f"📄 Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Test 3: Başarılı")
        else:
            print("❌ Test 3: Başarısız")
        
        # Test 4: Stok sorgusu
        print("\n📋 Test 4: Stok sorgusu")
        start_time = time.time()
        response = client.post(
            "/process_query/",
            json={"query": "Bu pantolonun S bedeni var mı?", "tenant_id": 1}
        )
        end_time = time.time()
        
        print(f"⏱️  Süre: {end_time - start_time:.2f} saniye")
        print(f"📊 Status: {response.status_code}")
        print(f"📄 Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Test 4: Başarılı")
        else:
            print("❌ Test 4: Başarısız")
        
        print("\n" + "=" * 50)
        print("🎉 Tüm testler tamamlandı!")
        print("=" * 50)
        
    except Exception as e:
        print(f"💥 Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_direct_tests() 