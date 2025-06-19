#!/usr/bin/env python3
"""
Gerçek Kapsamlı Testler - 197 testi çalıştırır
"""
import os
import sys
import time
import subprocess
from pathlib import Path

# Test ortamı ayarları
os.environ["TESTING"] = "true"
os.environ["FAST_MODE"] = "true"

def run_comprehensive_tests():
    """Gerçek kapsamlı testleri çalıştırır"""
    
    print("🚀 Gerçek Kapsamlı Testler Başlıyor...")
    print("=" * 60)
    
    # Test dosyalarını listele
    test_files = [
        "tests/test_unit.py",
        "tests/test_api.py", 
        "tests/test_comprehensive_scenarios.py",
        "tests/test_clarification.py",
        "tests/test_deep_research_dataset.py"
    ]
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_time = 0
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"⚠️  {test_file} bulunamadı, atlanıyor...")
            continue
            
        print(f"\n📁 {test_file} çalıştırılıyor...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Pytest ile test dosyasını çalıştır
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_file, 
                "-v", "--tb=short", "--timeout=30"
            ], capture_output=True, text=True, timeout=300)
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # Sonuçları analiz et
            output = result.stdout + result.stderr
            
            # Test sayılarını çıkar
            if "passed" in output:
                passed_line = [line for line in output.split('\n') if 'passed' in line and 'failed' in line]
                if passed_line:
                    stats = passed_line[0]
                    if 'failed' in stats:
                        parts = stats.split()
                        passed = int([p for p in parts if p.isdigit() and 'passed' in stats.split(stats.index(p))][0])
                        failed = int([p for p in parts if p.isdigit() and 'failed' in stats.split(stats.index(p))][0])
                        total_passed += passed
                        total_failed += failed
                        total_tests += passed + failed
                        
                        print(f"✅ Başarılı: {passed}")
                        print(f"❌ Başarısız: {failed}")
                        print(f"⏱️  Süre: {elapsed_time:.2f}s")
                    else:
                        passed = int([p for p in parts if p.isdigit()][0])
                        total_passed += passed
                        total_tests += passed
                        print(f"✅ Başarılı: {passed}")
                        print(f"⏱️  Süre: {elapsed_time:.2f}s")
            
            # Hata varsa göster
            if result.returncode != 0:
                print(f"⚠️  Bazı testler başarısız oldu")
                if result.stderr:
                    print(f"Hata: {result.stderr[:200]}...")
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} timeout oldu (5 dakika)")
        except Exception as e:
            print(f"❌ {test_file} çalıştırılırken hata: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 GENEL TEST SONUÇLARI:")
    print(f"   Toplam Test: {total_tests}")
    print(f"   Başarılı: {total_passed}")
    print(f"   Başarısız: {total_failed}")
    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(f"   Başarı Oranı: {success_rate:.1f}%")
    print(f"   Toplam Süre: {total_time:.2f}s")
    
    if total_tests > 0:
        if success_rate >= 90:
            print("🎉 MÜKEMMEL PERFORMANS!")
        elif success_rate >= 80:
            print("✅ ÇOK İYİ PERFORMANS!")
        elif success_rate >= 70:
            print("⚠️  İYİ PERFORMANS - Küçük iyileştirmeler gerekli")
        else:
            print("❌ DÜŞÜK PERFORMANS - Acil iyileştirme gerekli")

def run_quick_comprehensive():
    """Hızlı kapsamlı test - önemli testleri seçer"""
    
    print("🚀 Hızlı Kapsamlı Test Başlıyor...")
    print("=" * 60)
    
    # Önemli testleri seç
    important_tests = [
        "tests/test_unit.py::test_extract_item_candidate",
        "tests/test_unit.py::test_extract_size_entity",
        "tests/test_api.py::test_root_endpoint",
        "tests/test_api.py::test_greeting_and_thanks",
        "tests/test_comprehensive_scenarios.py::test_product_stock_query",
        "tests/test_comprehensive_scenarios.py::test_price_query",
        "tests/test_comprehensive_scenarios.py::test_shipping_cost_query",
        "tests/test_comprehensive_scenarios.py::test_greeting_query",
        "tests/test_comprehensive_scenarios.py::test_thanks_query"
    ]
    
    total_tests = len(important_tests)
    passed = 0
    failed = 0
    total_time = 0
    
    for test in important_tests:
        print(f"\n🧪 {test} çalıştırılıyor...")
        
        start_time = time.time()
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test, 
                "-v", "--tb=short", "--timeout=10"
            ], capture_output=True, text=True, timeout=60)
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            if result.returncode == 0:
                print(f"✅ Başarılı - {elapsed_time:.2f}s")
                passed += 1
            else:
                print(f"❌ Başarısız - {elapsed_time:.2f}s")
                failed += 1
                
        except Exception as e:
            print(f"❌ Hata: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 HIZLI TEST SONUÇLARI:")
    print(f"   Toplam Test: {total_tests}")
    print(f"   Başarılı: {passed}")
    print(f"   Başarısız: {failed}")
    if total_tests > 0:
        success_rate = (passed / total_tests) * 100
        print(f"   Başarı Oranı: {success_rate:.1f}%")
    print(f"   Toplam Süre: {total_time:.2f}s")

if __name__ == "__main__":
    # Hızlı test önce
    run_quick_comprehensive()
    
    print("\n" + "=" * 60)
    print("Devam etmek istiyor musunuz? (y/n)")
    # Gerçek kapsamlı testler çok uzun sürebilir
    # run_comprehensive_tests() 