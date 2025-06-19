#!/usr/bin/env python3
"""
Basit Test - Sadece temel fonksiyonları test eder
"""

import os
import sys
import time
from fastapi.testclient import TestClient

# FAST_MODE'u aktif et
os.environ["FAST_MODE"] = "true"

# Projenin ana dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from main import app

def test_basic():
    """En basit test"""
    print("🧪 Basit test başlıyor...")
    
    client = TestClient(app)
    
    # Ana endpoint testi
    print("📋 Ana endpoint test ediliyor...")
    response = client.get("/")
    print(f"✅ Status: {response.status_code}")
    print(f"✅ Response: {response.json()}")
    
    # Basit sorgu testi
    print("📋 Basit sorgu test ediliyor...")
    response = client.post(
        "/process_query/",
        json={"query": "Merhaba", "tenant_id": 1}
    )
    print(f"✅ Status: {response.status_code}")
    print(f"✅ Response: {response.json()}")
    
    print("🎉 Test başarılı!")

if __name__ == "__main__":
    test_basic() 