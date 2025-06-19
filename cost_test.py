#!/usr/bin/env python3
import asyncio
import aiohttp
import time
import random
import statistics

async def test_cost_analysis():
    base_url = 'http://localhost:8000'
    
    # Test sorguları
    queries = [
        'gecelik var mı',
        'pijama fiyatları', 
        'M beden var mı',
        'kargo ne kadar',
        'merhaba',
        'adresiniz nedir'
    ]
    
    results = []
    
    try:
        async with aiohttp.ClientSession() as session:
            # Health check
            async with session.get(f'{base_url}/health') as response:
                if response.status != 200:
                    print('❌ Sunucu çalışmıyor!')
                    return
            
            print('✅ Sunucu çalışıyor, testler başlıyor...')
            
            # 100 test sorgusu
            for i in range(100):
                query = random.choice(queries)
                start_time = time.time()
                
                payload = {
                    'query': query,
                    'tenant_id': 1,
                    'session_id': f'test_{i}'
                }
                
                async with session.post(f'{base_url}/process_query/', json=payload) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        results.append({
                            'query': query,
                            'response_time': response_time,
                            'intent': data.get('detected_intent', 'unknown'),
                            'method': data.get('nlu_method', 'unknown'),
                            'success': True
                        })
                    else:
                        results.append({
                            'query': query,
                            'response_time': response_time,
                            'success': False
                        })
                
                if i % 20 == 0:
                    print(f'Test {i+1}/100 tamamlandı...')
    
    except Exception as e:
        print(f'❌ Test hatası: {e}')
        return
    
    # Analiz
    successful = [r for r in results if r['success']]
    response_times = [r['response_time'] for r in successful]
    
    if successful:
        avg_time = statistics.mean(response_times)
        success_rate = len(successful) / len(results) * 100
        
        print(f'\n📊 TEST SONUÇLARI:')
        print(f'   Toplam test: {len(results)}')
        print(f'   Başarılı: {len(successful)}')
        print(f'   Başarı oranı: {success_rate:.1f}%')
        print(f'   Ortalama yanıt süresi: {avg_time*1000:.0f}ms')
        print(f'   En hızlı: {min(response_times)*1000:.0f}ms')
        print(f'   En yavaş: {max(response_times)*1000:.0f}ms')
        
        # 1M sorgu maliyet hesabı
        monthly_queries = 1_000_000
        successful_monthly = int(monthly_queries * success_rate / 100)
        
        # Varsayımsal maliyetler
        server_cost = 200  # Aylık sunucu
        database_cost = 50  # Aylık DB
        support_cost = 150  # Aylık destek
        cpu_cost_per_query = 0.0001  # CPU/sorgu
        
        fixed_costs = server_cost + database_cost + support_cost
        variable_costs = successful_monthly * cpu_cost_per_query
        total_monthly = fixed_costs + variable_costs
        
        print(f'\n💰 1 MİLYON SORGU/AY MALİYET:')
        print(f'   Sabit maliyetler: ₺{fixed_costs}/ay')
        print(f'   Değişken maliyetler: ₺{variable_costs:.2f}/ay')
        print(f'   Toplam aylık: ₺{total_monthly:.2f}')
        print(f'   Sorgu başına: ₺{total_monthly/successful_monthly:.6f}')
        print(f'   Yıllık toplam: ₺{total_monthly*12:,.2f}')
        
        # Intent dağılımı
        intents = {}
        for r in successful:
            intent = r.get('intent', 'unknown')
            intents[intent] = intents.get(intent, 0) + 1
        
        print(f'\n🎯 INTENT DAĞILIMI:')
        for intent, count in sorted(intents.items(), key=lambda x: x[1], reverse=True):
            print(f'   {intent}: {count} ({count/len(successful)*100:.1f}%)')
    
    else:
        print('❌ Hiç başarılı test yok!')

if __name__ == "__main__":
    asyncio.run(test_cost_analysis()) 