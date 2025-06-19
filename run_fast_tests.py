#!/usr/bin/env python3
"""
Hızlı Test Runner - FAST_MODE ile
Bu script testleri hızlı modda çalıştırır.
"""

import os
import sys
import subprocess
import time

def run_fast_tests():
    """FAST_MODE ile testleri çalıştır"""
    
    # FAST_MODE'u aktif et
    os.environ["FAST_MODE"] = "true"
    
    print("🚀 Hızlı Test Modu Başlatılıyor...")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test komutları
    test_commands = [
        ["python", "-m", "pytest", "test_optimized.py", "-v", "--tb=short"],
        ["python", "-m", "pytest", "tests/test_unit.py", "-v", "--tb=short"],
        ["python", "-m", "pytest", "tests/test_api_fixed.py", "-v", "--tb=short", "-k", "test_root_endpoint"],
    ]
    
    results = []
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n📋 Test {i}/{len(test_commands)}: {' '.join(cmd)}")
        print("-" * 40)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 saniye timeout
            )
            
            if result.returncode == 0:
                print("✅ Başarılı")
                print(result.stdout[-500:])  # Son 500 karakter
            else:
                print("❌ Başarısız")
                print(result.stderr[-500:])  # Son 500 karakter
            
            results.append((cmd, result.returncode == 0))
            
        except subprocess.TimeoutExpired:
            print("⏰ Timeout - Test çok uzun sürdü")
            results.append((cmd, False))
        except Exception as e:
            print(f"💥 Hata: {e}")
            results.append((cmd, False))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 50)
    print("📊 Test Sonuçları:")
    print("=" * 50)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for cmd, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {' '.join(cmd)}")
    
    print(f"\n⏱️  Toplam Süre: {total_time:.2f} saniye")
    print(f"📈 Başarı Oranı: {successful}/{total} ({successful/total*100:.1f}%)")
    
    if successful == total:
        print("🎉 Tüm testler başarılı!")
    else:
        print("⚠️  Bazı testler başarısız oldu.")

if __name__ == "__main__":
    run_fast_tests() 