#!/usr/bin/env python3
# test_simple.py - Basit test dosyası

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_basic_imports():
    """Temel importları test et"""
    try:
        import fastapi
        print("✓ FastAPI import edildi")
    except ImportError as e:
        print(f"✗ FastAPI import hatası: {e}")
        return False
    
    try:
        import pydantic
        print("✓ Pydantic import edildi")
    except ImportError as e:
        print(f"✗ Pydantic import hatası: {e}")
        return False
    
    try:
        import zeyrek
        print("✓ Zeyrek import edildi")
    except ImportError as e:
        print(f"✗ Zeyrek import hatası: {e}")
        return False
    
    try:
        import fasttext
        print("✓ FastText import edildi")
    except ImportError as e:
        print(f"✗ FastText import hatası: {e}")
        return False
    
    return True

def test_database():
    """Veritabanı bağlantısını test et"""
    try:
        import sqlite3
        conn = sqlite3.connect("chatbot_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"✓ Veritabanı bağlantısı başarılı. Tablolar: {[t[0] for t in tables]}")
        conn.close()
        return True
    except Exception as e:
        print(f"✗ Veritabanı hatası: {e}")
        return False

def test_basic_functions():
    """Temel fonksiyonları test et (modelleri yüklemeden)"""
    try:
        # Sadece fonksiyon tanımlarını kontrol et
        from main import extract_simple_entities, _preprocess_text_for_matching
        print("✓ Temel fonksiyonlar import edildi")
        return True
    except Exception as e:
        print(f"✗ Fonksiyon import hatası: {e}")
        return False

if __name__ == "__main__":
    print("=== BASİT TEST BAŞLIYOR ===")
    
    success = True
    success &= test_basic_imports()
    success &= test_database()
    success &= test_basic_functions()
    
    print("\n=== TEST SONUCU ===")
    if success:
        print("✓ Tüm temel testler başarılı!")
    else:
        print("✗ Bazı testler başarısız!")
    
    sys.exit(0 if success else 1) 