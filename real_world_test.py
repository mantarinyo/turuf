#!/usr/bin/env python3
"""
Gerçek Dünya Testi - Bozuk cümleler, yazım hataları, edge case'ler
"""

import asyncio
import aiohttp
import time
import random

class RealWorldTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.tenant_id = 1
        
        # Gerçek kullanıcı soruları - bozuk, hatalı, eksik
        self.broken_queries = {
            "ürün_arama_bozuk": [
                "gecelik varmı",  # Yazım hatası
                "pijama var mi acaba?",  # Boşluk hatası
                "abiye elbise vr mı",  # Kısaltma
                "tesettür giyim varmı sizde",  # Karışık
                "günlük elbise",  # Eksik soru
                "kışlık mont var",  # Eksik soru eki
                "geclik varmi",  # Çoklu yazım hatası
                "pjama takımı",  # Harf eksik
                "abiye elbse var mı",  # Harf karışık
                "gecelik vr",  # Çok kısaltılmış
                "pijama???",  # Fazla noktalama
                "GECELIK VAR MI",  # BÜYÜK HARF
                "gecelik    var     mı",  # Fazla boşluk
                "gecelik\nvar\nmı",  # Satır sonu
                "geelik var mı",  # Typo
                "pijama takım var mı",  # Kelime eksik
                "abiye elbisesi var mı",  # Ek hata
                "tesettür giyim mevcut mu",  # Farklı ifade
                "günlük elbise satıyor musunuz",  # Uzun form
                "kışlık mont bulunur mu",  # Formal
                "gecelik arıyorum",  # Farklı yapı
                "pijama takımı lazım",  # İhtiyaç ifadesi
            ],
            "fiyat_sorgulama_bozuk": [
                "bu elbisenin fiyatı ne",  # Eksik
                "gecelik fiyat nedir",  # Kelime eksik
                "pijama kaça",  # Çok kısa
                "abiye elbise ne kadar eder",  # Uzun
                "mont fiyatı kaç",  # Eksik
                "ayakkabı kaça patlar",  # Argo
                "çanta fiyat neydi",  # Geçmiş zaman
                "takı ne kadar tutar",  # Farklı fiil
                "eşarp kaç para",  # Basit
                "trençkot fiyatı nedir acaba",  # Uzun
                "pantolon kaça geliyor",  # Argo
                "bluz fiyat bilgisi",  # Eksik yapı
                "fyat ne",  # Typo
                "fiyatı kaç tl",  # Kısaltma
                "kaça satıyorsunuz",  # Genel soru
                "ne kadar istiyorsunuz",  # Belirsiz
                "ücret nedir",  # Farklı kelime
                "para ne kadar",  # Basit
                "maliyet nedir",  # Teknik terim
                "değeri ne",  # Farklı anlam
            ],
            "stok_sorgulama_bozuk": [
                "M beden varmı",  # Yazım hatası
                "L beden mevcut",  # Eksik soru
                "38 numara var",  # Eksik ek
                "bu renk stokta",  # Eksik soru
                "XL beden kaldımı",  # Yazım hatası
                "S beden vr mı",  # Kısaltma
                "40 numara mevcut mu acaba",  # Uzun
                "büyük beden var mıdır",  # Formal
                "siyah renk varmı",  # Yazım hatası
                "beyaz modeli var",  # Eksik soru
                "stok var mı",  # Çok genel
                "elimizde var mı",  # Belirsiz
                "mevcut mu",  # Çok kısa
                "kaldı mı",  # Çok kısa
                "satışta mı",  # Farklı ifade
                "bulunur mu",  # Formal
                "tükenmiş mi",  # Olumsuz
                "hangi bedenler var",  # Açık uçlu
                "hangi renkler mevcut",  # Açık uçlu
                "ne var",  # Çok genel
            ],
            "edge_cases": [
                "",  # Boş
                " ",  # Sadece boşluk
                "a",  # Tek harf
                "???",  # Sadece noktalama
                "123",  # Sadece rakam
                "asdfghjkl",  # Random
                "gecelik var mı gecelik var mı",  # Tekrar
                "gecelik pijama abiye elbise mont",  # Karma
                "var mı var mı var mı",  # Tekrar kelime
                "ne ne ne ne",  # Tekrar
                "merhaba gecelik var mı teşekkürler",  # Karma intent
                "gecelik var mı fiyatı ne kadar",  # Çoklu soru
                "GECEEEELIK VAR MIIIIII",  # Abartılı
                "gecelik... var... mı...",  # Noktalı
                "gecelik/var/mı",  # Slash
                "gecelik&var&mı",  # Ampersand
                "gecelik+var+mı",  # Plus
                "gece lik var mı",  # Yanlış ayrım
                "geceli k varmı",  # Karışık ayrım
                "😀 gecelik var mı 😀",  # Emoji
                "Gecelik Var Mı?",  # Title Case
            ],
            "kargo_bozuk": [
                "kargo ne kadar sürer acaba",
                "kargo ücreti nedir tam olarak",
                "ücretsiz kargo varmı",
                "hızlı kargo seçeneği var",
                "aynı gün teslimat yapıyor musunuz",
                "kargo takibi nasıl",
                "yurtdışı kargo gönderiyor musunuz",
                "kargo firması hangisi",
                "kargo kaç günde gelir",
                "kargo bedava mı",
            ],
            "selamlama_bozuk": [
                "mrb",
                "slm",
                "sa",
                "iyi günler",
                "günaydın",
                "merhaba nasılsınız",
                "selam",
                "hey",
                "hii",
                "hello",
            ]
        }
    
    async def test_single_query(self, session: aiohttp.ClientSession, query: str, expected_intent: str = None) -> dict:
        """Tek sorgu testi"""
        start_time = time.time()
        
        try:
            payload = {
                "query": query,
                "tenant_id": self.tenant_id,
                "session_id": f"real_test_{random.randint(1000, 9999)}"
            }
            
            async with session.post(f"{self.base_url}/process_query/", json=payload) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    detected_intent = data.get("detected_intent", "unknown")
                    method = data.get("nlu_method", "unknown")
                    bot_response = data.get("bot_response", "")
                    
                    # Intent doğruluğu kontrolü
                    correct = True
                    if expected_intent and expected_intent not in detected_intent:
                        correct = False
                    
                    return {
                        "query": query,
                        "expected": expected_intent,
                        "detected": detected_intent,
                        "method": method,
                        "response_time": response_time,
                        "bot_response": bot_response,
                        "correct": correct,
                        "success": True
                    }
                else:
                    return {
                        "query": query,
                        "expected": expected_intent,
                        "detected": "error",
                        "method": "error",
                        "response_time": response_time,
                        "bot_response": f"HTTP {response.status}",
                        "correct": False,
                        "success": False
                    }
                    
        except Exception as e:
            return {
                "query": query,
                "expected": expected_intent,
                "detected": "error",
                "method": "error",
                "response_time": time.time() - start_time,
                "bot_response": str(e),
                "correct": False,
                "success": False
            }
    
    async def run_comprehensive_real_world_test(self):
        """Kapsamlı gerçek dünya testi"""
        
        print("🌍 GERÇEK DÜNYA KAPSAMLI TESTİ BAŞLIYOR...")
        print("="*80)
        
        all_results = []
        total_tests = 0
        
        async with aiohttp.ClientSession() as session:
            # Her kategori için test
            for category, queries in self.broken_queries.items():
                expected_intent = category.replace("_bozuk", "").replace("_", "_")
                
                print(f"\n🧪 {category.upper()} kategorisi test ediliyor...")
                
                category_results = []
                
                for query in queries:
                    result = await self.test_single_query(session, query, expected_intent)
                    category_results.append(result)
                    all_results.append(result)
                    total_tests += 1
                
                # Kategori analizi
                successful = [r for r in category_results if r["success"]]
                correct = [r for r in category_results if r["correct"]]
                
                if successful:
                    avg_time = sum(r["response_time"] for r in successful) / len(successful)
                    success_rate = len(successful) / len(category_results) * 100
                    accuracy_rate = len(correct) / len(successful) * 100 if successful else 0
                    
                    print(f"   ✅ Başarı: {len(successful)}/{len(category_results)} ({success_rate:.1f}%)")
                    print(f"   🎯 Doğruluk: {len(correct)}/{len(successful)} ({accuracy_rate:.1f}%)")
                    print(f"   ⚡ Ortalama süre: {avg_time*1000:.0f}ms")
                    
                    # Hatalı örnekler
                    wrong = [r for r in successful if not r["correct"]]
                    if wrong:
                        print(f"   ❌ Yanlış anlaşılan örnekler:")
                        for w in wrong[:3]:  # İlk 3 hatayı göster
                            print(f"      '{w['query']}' → {w['detected']} (beklenen: {w['expected']})")
        
        # Genel analiz
        print(f"\n" + "="*80)
        print(f"📊 GENEL SONUÇLAR")
        print(f"="*80)
        
        successful_results = [r for r in all_results if r["success"]]
        correct_results = [r for r in all_results if r["correct"]]
        
        if successful_results:
            overall_success = len(successful_results) / total_tests * 100
            overall_accuracy = len(correct_results) / len(successful_results) * 100
            avg_response_time = sum(r["response_time"] for r in successful_results) / len(successful_results)
            
            print(f"Toplam Test: {total_tests}")
            print(f"Başarılı İstek: {len(successful_results)} ({overall_success:.1f}%)")
            print(f"Doğru Intent: {len(correct_results)} ({overall_accuracy:.1f}%)")
            print(f"Ortalama Yanıt Süresi: {avg_response_time*1000:.0f}ms")
            
            # Method dağılımı
            methods = {}
            for r in successful_results:
                method = r["method"]
                methods[method] = methods.get(method, 0) + 1
            
            print(f"\n🔧 METHOD DAĞILIMI:")
            for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(successful_results) * 100
                print(f"   {method}: {count} ({percentage:.1f}%)")
            
            # En çok hata yapılan sorgular
            wrong_results = [r for r in successful_results if not r["correct"]]
            if wrong_results:
                print(f"\n❌ EN PROBLEMLI SORGULAR:")
                for w in wrong_results[:10]:
                    print(f"   '{w['query']}' → {w['detected']} (beklenen: {w['expected']})")
        
        return all_results

async def main():
    """Ana test fonksiyonu"""
    tester = RealWorldTester()
    
    # Sunucu kontrolü
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{tester.base_url}/health") as response:
                if response.status != 200:
                    print("❌ Sunucu çalışmıyor!")
                    return
    except:
        print("❌ Sunucuya bağlanılamıyor!")
        return
    
    print("✅ Sunucu çalışıyor, gerçek dünya testleri başlıyor...")
    await tester.run_comprehensive_real_world_test()

if __name__ == "__main__":
    asyncio.run(main()) 