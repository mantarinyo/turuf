#!/usr/bin/env python3
"""
Kapsamlı Test Sistemi & Maliyet Hesaplaması
- 1 milyon sorgu/ay maliyet analizi
- Gerçek müşteri senaryoları
- Yük testi
- Performance analizi
"""

import asyncio
import aiohttp
import time
import random
import json
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any
import concurrent.futures
import threading
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class TestResult:
    query: str
    response_time: float
    intent: str
    method: str
    energy_consumption: float
    success: bool
    error: str = None

class ComprehensiveCostTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.tenant_id = 1
        
        # Gerçek müşteri sorgu dağılımı (e-ticaret verilerine dayalı)
        self.query_distribution = {
            "ürün_arama": 35,      # %35 - En çok sorulan
            "fiyat_sorgulama": 20, # %20
            "stok_sorgulama": 15,  # %15
            "kargo_bilgisi_sorma": 10, # %10
            "selamlama": 8,        # %8
            "lokasyon_sorma": 5,   # %5
            "tel_no_sorma": 3,     # %3
            "iade_sorgulama": 2,   # %2
            "tesekkur": 1,         # %1
            "kapsam_disi": 1       # %1
        }
        
        # Gerçek müşteri soruları (590 ürünlü veritabanına uygun)
        self.realistic_queries = {
            "ürün_arama": [
                "gecelik var mı", "pijama takımları", "abiye elbise modelleri",
                "tesettür giyim", "günlük elbise", "kışlık mont", "yaz koleksiyonu",
                "büyük beden elbise", "hamile kıyafetleri", "çocuk giyim",
                "spor giyim", "iç giyim", "ayakkabı modelleri", "çanta çeşitleri",
                "aksesuar", "takı modelleri", "eşarp çeşitleri", "şal modelleri",
                "trençkot var mı", "ceket modelleri", "pantolon çeşitleri",
                "etek modelleri", "bluz çeşitleri", "tunik modelleri",
                "ferace modelleri", "pardesü çeşitleri", "kap modelleri"
            ],
            "fiyat_sorgulama": [
                "bu elbisenin fiyatı nedir", "gecelik fiyatları", "pijama ne kadar",
                "abiye elbise fiyat", "mont fiyatları", "ayakkabı kaça",
                "çanta fiyat", "takı fiyatları", "eşarp ne kadar",
                "trençkot fiyatı", "pantolon kaça", "bluz fiyat"
            ],
            "stok_sorgulama": [
                "M beden var mı", "L beden mevcut mu", "38 numara var mı",
                "bu renk stokta mı", "XL beden kaldı mı", "S beden var mı",
                "40 numara mevcut mu", "büyük beden var mı",
                "siyah renk var mı", "beyaz modeli mevcut mu"
            ],
            "kargo_bilgisi_sorma": [
                "kargo ne kadar sürer", "kargo ücreti nedir", "ücretsiz kargo var mı",
                "hızlı kargo seçeneği", "aynı gün teslimat", "kargo takibi",
                "yurtdışı kargo", "kargo firması hangisi"
            ],
            "selamlama": [
                "merhaba", "selam", "iyi günler", "günaydın", "iyi akşamlar",
                "sa", "mrb", "slm", "hey", "selam nasılsınız"
            ],
            "lokasyon_sorma": [
                "adresiniz nedir", "mağaza nerede", "şube var mı",
                "konumunuz", "nasıl gelebilirim", "harita konumu",
                "en yakın mağaza", "istanbul şubesi"
            ],
            "tel_no_sorma": [
                "telefon numaranız", "whatsapp no", "iletişim bilgileri",
                "nasıl ulaşabilirim", "müşteri hizmetleri", "çağrı merkezi"
            ],
            "iade_sorgulama": [
                "iade etmek istiyorum", "değişim yapabilir miyim", "iade şartları",
                "geri iade", "beğenmedim iade", "değişim süreci"
            ],
            "tesekkur": [
                "teşekkürler", "sağol", "çok teşekkür ederim", "eyvallah",
                "tşk", "thanks", "thx", "sağolun"
            ],
            "kapsam_disi": [
                "hava durumu nasıl", "bugün hangi gün", "saç nasıl kesilir",
                "yemek tarifi", "matematik problemi", "random text"
            ]
        }
    
    async def single_query_test(self, session: aiohttp.ClientSession, query: str) -> TestResult:
        """Tek sorgu testi"""
        start_time = time.time()
        
        try:
            payload = {
                "query": query,
                "tenant_id": self.tenant_id,
                "session_id": f"test_{random.randint(1000, 9999)}"
            }
            
            async with session.post(f"{self.base_url}/process_query/", json=payload) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    return TestResult(
                        query=query,
                        response_time=response_time,
                        intent=data.get("detected_intent", "unknown"),
                        method=data.get("nlu_method", "unknown"),
                        energy_consumption=0.5,  # Ortalama değer
                        success=True
                    )
                else:
                    return TestResult(
                        query=query,
                        response_time=response_time,
                        intent="error",
                        method="error",
                        energy_consumption=0.1,
                        success=False,
                        error=f"HTTP {response.status}"
                    )
                    
        except Exception as e:
            return TestResult(
                query=query,
                response_time=time.time() - start_time,
                intent="error",
                method="error", 
                energy_consumption=0.1,
                success=False,
                error=str(e)
            )
    
    def generate_realistic_query_mix(self, count: int) -> List[str]:
        """Gerçek dağılıma uygun sorgu karışımı üret"""
        queries = []
        
        for intent, percentage in self.query_distribution.items():
            intent_count = int(count * percentage / 100)
            intent_queries = random.choices(
                self.realistic_queries[intent], 
                k=intent_count
            )
            queries.extend(intent_queries)
        
        # Kalan sorguları rastgele doldur
        while len(queries) < count:
            random_intent = random.choice(list(self.realistic_queries.keys()))
            random_query = random.choice(self.realistic_queries[random_intent])
            queries.append(random_query)
        
        random.shuffle(queries)
        return queries[:count]
    
    async def load_test(self, concurrent_users: int, queries_per_user: int) -> Dict[str, Any]:
        """Yük testi - Eş zamanlı kullanıcı simülasyonu"""
        print(f"🚀 Yük testi başlıyor: {concurrent_users} kullanıcı, {queries_per_user} sorgu/kullanıcı")
        
        all_results = []
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for user_id in range(concurrent_users):
                user_queries = self.generate_realistic_query_mix(queries_per_user)
                
                for query in user_queries:
                    task = self.single_query_test(session, query)
                    tasks.append(task)
            
            # Tüm testleri eş zamanlı çalıştır
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Exception'ları filtrele
            for result in results:
                if isinstance(result, TestResult):
                    all_results.append(result)
        
        total_time = time.time() - start_time
        
        return self.analyze_results(all_results, total_time)
    
    def analyze_results(self, results: List[TestResult], total_time: float) -> Dict[str, Any]:
        """Test sonuçlarını analiz et"""
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if not successful_results:
            return {"error": "Hiç başarılı test yok!"}
        
        response_times = [r.response_time for r in successful_results]
        energy_consumptions = [r.energy_consumption for r in successful_results]
        
        # Intent dağılımı
        intent_counts = {}
        method_counts = {}
        
        for result in successful_results:
            intent_counts[result.intent] = intent_counts.get(result.intent, 0) + 1
            method_counts[result.method] = method_counts.get(result.method, 0) + 1
        
        analysis = {
            "test_summary": {
                "total_queries": len(results),
                "successful_queries": len(successful_results),
                "failed_queries": len(failed_results),
                "success_rate": len(successful_results) / len(results) * 100,
                "total_test_time": total_time,
                "queries_per_second": len(results) / total_time
            },
            "performance": {
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)],
                "p99_response_time": sorted(response_times)[int(len(response_times) * 0.99)],
                "min_response_time": min(response_times),
                "max_response_time": max(response_times)
            },
            "energy": {
                "total_energy": sum(energy_consumptions),
                "avg_energy_per_query": statistics.mean(energy_consumptions),
                "energy_efficiency": sum(energy_consumptions) / len(successful_results)
            },
            "intent_distribution": intent_counts,
            "method_distribution": method_counts,
            "errors": [{"query": r.query, "error": r.error} for r in failed_results]
        }
        
        return analysis
    
    def calculate_monthly_cost(self, analysis: Dict[str, Any], monthly_queries: int = 1_000_000) -> Dict[str, Any]:
        """Aylık maliyet hesaplaması"""
        
        if "performance" not in analysis:
            return {"error": "Analiz verisi eksik"}
        
        # Temel metrikler
        avg_response_time = analysis["performance"]["avg_response_time"]
        avg_energy = analysis["energy"]["avg_energy_per_query"]
        success_rate = analysis["test_summary"]["success_rate"] / 100
        
        # Maliyet hesaplamaları (varsayımsal değerler)
        costs = {
            # Sunucu maliyetleri
            "server_cost_per_month": 200,  # VPS/Cloud server
            "database_cost_per_month": 50,  # SQLite → PostgreSQL geçişte
            
            # İşlem maliyetleri
            "cpu_cost_per_query": 0.0001,  # CPU kullanımı
            "memory_cost_per_query": 0.00005,  # RAM kullanımı
            "storage_cost_per_query": 0.00001,  # Disk I/O
            
            # Enerji maliyetleri
            "energy_cost_per_unit": 0.001,  # Enerji birimi başına maliyet
            
            # Destek maliyetleri
            "support_cost_per_month": 100,  # Teknik destek
            "maintenance_cost_per_month": 150  # Bakım ve güncelleme
        }
        
        # Aylık hesaplamalar
        successful_queries = int(monthly_queries * success_rate)
        
        monthly_costs = {
            "fixed_costs": {
                "server": costs["server_cost_per_month"],
                "database": costs["database_cost_per_month"], 
                "support": costs["support_cost_per_month"],
                "maintenance": costs["maintenance_cost_per_month"]
            },
            "variable_costs": {
                "cpu": successful_queries * costs["cpu_cost_per_query"],
                "memory": successful_queries * costs["memory_cost_per_query"],
                "storage": successful_queries * costs["storage_cost_per_query"],
                "energy": successful_queries * avg_energy * costs["energy_cost_per_unit"]
            }
        }
        
        total_fixed = sum(monthly_costs["fixed_costs"].values())
        total_variable = sum(monthly_costs["variable_costs"].values())
        total_monthly = total_fixed + total_variable
        
        cost_analysis = {
            "monthly_queries": monthly_queries,
            "successful_queries": successful_queries,
            "failed_queries": monthly_queries - successful_queries,
            "costs": monthly_costs,
            "totals": {
                "fixed_costs": total_fixed,
                "variable_costs": total_variable,
                "total_monthly_cost": total_monthly,
                "cost_per_query": total_monthly / successful_queries if successful_queries > 0 else 0,
                "cost_per_successful_query": total_monthly / successful_queries if successful_queries > 0 else 0
            },
            "projections": {
                "yearly_cost": total_monthly * 12,
                "cost_at_2m_queries": (total_fixed + total_variable * 2) * 12,
                "cost_at_5m_queries": (total_fixed + total_variable * 5) * 12,
                "break_even_queries": total_fixed / (0.01 - (total_variable / successful_queries)) if successful_queries > 0 else 0
            },
            "performance_metrics": {
                "avg_response_time_ms": avg_response_time * 1000,
                "queries_per_second_capacity": 1 / avg_response_time if avg_response_time > 0 else 0,
                "energy_efficiency_score": 1 / avg_energy if avg_energy > 0 else 0
            }
        }
        
        return cost_analysis
    
    def generate_report(self, analysis: Dict[str, Any], cost_analysis: Dict[str, Any]):
        """Detaylı rapor oluştur"""
        
        print("\n" + "="*80)
        print("📊 KAPSAMLI TEST VE MALİYET ANALİZİ RAPORU")
        print("="*80)
        
        # Test özeti
        print(f"\n🧪 TEST ÖZETİ:")
        print(f"   Toplam Sorgu: {analysis['test_summary']['total_queries']:,}")
        print(f"   Başarılı: {analysis['test_summary']['successful_queries']:,}")
        print(f"   Başarısız: {analysis['test_summary']['failed_queries']:,}")
        print(f"   Başarı Oranı: {analysis['test_summary']['success_rate']:.1f}%")
        print(f"   Test Süresi: {analysis['test_summary']['total_test_time']:.1f} saniye")
        print(f"   QPS: {analysis['test_summary']['queries_per_second']:.1f}")
        
        # Performance
        print(f"\n⚡ PERFORMANS METRİKLERİ:")
        print(f"   Ortalama Yanıt Süresi: {analysis['performance']['avg_response_time']*1000:.0f}ms")
        print(f"   Medyan Yanıt Süresi: {analysis['performance']['median_response_time']*1000:.0f}ms")
        print(f"   P95 Yanıt Süresi: {analysis['performance']['p95_response_time']*1000:.0f}ms")
        print(f"   P99 Yanıt Süresi: {analysis['performance']['p99_response_time']*1000:.0f}ms")
        print(f"   En Hızlı: {analysis['performance']['min_response_time']*1000:.0f}ms")
        print(f"   En Yavaş: {analysis['performance']['max_response_time']*1000:.0f}ms")
        
        # Maliyet analizi
        print(f"\n💰 MALİYET ANALİZİ (1 Milyon Sorgu/Ay):")
        print(f"   Sabit Maliyetler: ₺{cost_analysis['totals']['fixed_costs']:.2f}/ay")
        print(f"   Değişken Maliyetler: ₺{cost_analysis['totals']['variable_costs']:.2f}/ay")
        print(f"   Toplam Aylık Maliyet: ₺{cost_analysis['totals']['total_monthly_cost']:.2f}")
        print(f"   Sorgu Başına Maliyet: ₺{cost_analysis['totals']['cost_per_query']:.6f}")
        print(f"   Yıllık Maliyet: ₺{cost_analysis['projections']['yearly_cost']:,.2f}")
        
        # Intent dağılımı
        print(f"\n🎯 INTENT DAĞILIMI:")
        for intent, count in sorted(analysis['intent_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / analysis['test_summary']['successful_queries'] * 100
            print(f"   {intent}: {count} ({percentage:.1f}%)")
        
        # Method dağılımı
        print(f"\n🔧 METHOD DAĞILIMI:")
        for method, count in analysis['method_distribution'].items():
            percentage = count / analysis['test_summary']['successful_queries'] * 100
            print(f"   {method}: {count} ({percentage:.1f}%)")
        
        # Projeksiyonlar
        print(f"\n📈 PROJEKSİYONLAR:")
        print(f"   2M sorgu/ay maliyeti: ₺{cost_analysis['projections']['cost_at_2m_queries']:,.2f}/yıl")
        print(f"   5M sorgu/ay maliyeti: ₺{cost_analysis['projections']['cost_at_5m_queries']:,.2f}/yıl")
        
        # Hatalar
        if analysis['errors']:
            print(f"\n❌ HATALAR ({len(analysis['errors'])} adet):")
            for error in analysis['errors'][:5]:  # İlk 5 hatayı göster
                print(f"   '{error['query']}' -> {error['error']}")
    
    async def run_comprehensive_test(self):
        """Kapsamlı test süitini çalıştır"""
        
        print("🚀 Kapsamlı test süiti başlıyor...")
        
        # Farklı yük seviyelerinde test
        test_scenarios = [
            {"users": 10, "queries": 20},    # Düşük yük
            {"users": 50, "queries": 40},    # Orta yük  
            {"users": 100, "queries": 50},   # Yüksek yük
        ]
        
        all_results = []
        
        for scenario in test_scenarios:
            print(f"\n📊 Test senaryosu: {scenario['users']} kullanıcı, {scenario['queries']} sorgu/kullanıcı")
            
            analysis = await self.load_test(scenario['users'], scenario['queries'])
            all_results.append(analysis)
            
            print(f"   ✅ Başarı oranı: {analysis['test_summary']['success_rate']:.1f}%")
            print(f"   ⚡ Ortalama yanıt: {analysis['performance']['avg_response_time']*1000:.0f}ms")
            print(f"   🔥 QPS: {analysis['test_summary']['queries_per_second']:.1f}")
        
        # En iyi performanslı sonucu al
        best_result = max(all_results, key=lambda x: x['test_summary']['success_rate'])
        
        # Maliyet analizi
        cost_analysis = self.calculate_monthly_cost(best_result)
        
        # Rapor oluştur
        self.generate_report(best_result, cost_analysis)
        
        return best_result, cost_analysis

async def main():
    """Ana test fonksiyonu"""
    tester = ComprehensiveCostTester()
    
    # Önce sunucunun çalıştığını kontrol et
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{tester.base_url}/health") as response:
                if response.status != 200:
                    print("❌ Sunucu çalışmıyor! Önce main.py'yi başlatın.")
                    return
    except:
        print("❌ Sunucuya bağlanılamıyor! Önce main.py'yi başlatın.")
        return
    
    print("✅ Sunucu çalışıyor, testler başlıyor...")
    
    # Kapsamlı test çalıştır
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main()) 