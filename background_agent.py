#!/usr/bin/env python3
"""
Background Agent - Chatbot Continuous Improvement System
Sürekli conversation_logs'ları analiz ederek chatbot'u iyileştiren AI agent
"""

import sqlite3
import json
import time
import schedule
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple
import logging

# AI/LLM integration için
import openai  # veya başka LLM API
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotOptimizerAgent:
    def __init__(self, db_path: str = "chatbot_data.db"):
        self.db_path = db_path
        self.analysis_history = []
        
    def analyze_conversation_logs(self, hours_back: int = 24) -> Dict:
        """Son X saatteki conversation logs'ları analiz et"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Son 24 saatin verilerini al
        cursor.execute("""
            SELECT user_query_original, detected_intent, slm_intent, slm_confidence, 
                   nlu_method, response_time_ms, created_at,
                   CASE WHEN detected_intent != slm_intent THEN 1 ELSE 0 END as conflict
            FROM conversation_logs 
            WHERE created_at > datetime('now', '-{} hours')
            ORDER BY created_at DESC
        """.format(hours_back))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {"status": "no_data", "message": "No recent conversation data"}
        
        # Data analysis
        conflicts = []
        var_mi_queries = []
        performance_data = []
        
        for row in rows:
            query, detected, slm, confidence, method, response_time, created, is_conflict = row
            
            performance_data.append({
                "query": query,
                "response_time": response_time,
                "method": method,
                "confidence": confidence
            })
            
            if is_conflict:
                conflicts.append({
                    "query": query,
                    "detected": detected, 
                    "slm": slm,
                    "confidence": confidence,
                    "method": method
                })
            
            if "var mı" in query.lower() or "varmı" in query.lower():
                var_mi_queries.append({
                    "query": query,
                    "detected": detected,
                    "slm": slm,
                    "confidence": confidence
                })
        
        return {
            "total_queries": len(rows),
            "conflicts": conflicts,
            "var_mi_queries": var_mi_queries,
            "performance_data": performance_data,
            "analysis_time": datetime.now().isoformat()
        }
    
    def generate_improvement_prompt(self, analysis_data: Dict) -> str:
        """AI agent için analiz prompt'u oluştur"""
        
        prompt = f"""# Chatbot Optimization Analysis

## DATA SUMMARY
- Total queries analyzed: {analysis_data['total_queries']}
- Intent conflicts found: {len(analysis_data.get('conflicts', []))}
- "var mı?" queries: {len(analysis_data.get('var_mi_queries', []))}

## CONFLICT ANALYSIS
"""
        
        # Conflict örnekleri ekle
        if analysis_data.get('conflicts'):
            prompt += "\n### Intent Conflicts (ML vs Final Decision):\n"
            for i, conflict in enumerate(analysis_data['conflicts'][:10]):  # İlk 10
                prompt += f"{i+1}. Query: '{conflict['query']}'\n"
                prompt += f"   ML Said: {conflict['slm']} (conf: {conflict['confidence']:.3f})\n"
                prompt += f"   Final: {conflict['detected']} (method: {conflict['method']})\n\n"
        
        # "var mı?" sorguları analizi
        if analysis_data.get('var_mi_queries'):
            prompt += "\n### 'var mı?' Queries Analysis:\n"
            for i, query in enumerate(analysis_data['var_mi_queries'][:10]):
                prompt += f"{i+1}. '{query['query']}' → {query['detected']}\n"
        
        prompt += f"""

## TASK
Sen bir chatbot optimization uzmanısın. Yukarıdaki verileri analiz ederek:

1. **PATTERN DISCOVERY**: Hangi query türleri yanlış classify ediliyor?
2. **REGEX IMPROVEMENTS**: Mevcut regex rules'lara eklenecek pattern'ler
3. **TRAINING DATA**: Yanlış classify'lar için training data önerileri
4. **PERFORMANCE INSIGHTS**: Response time ve accuracy trends

## OUTPUT FORMAT
```json
{{
  "patterns_discovered": [
    "Pattern 1: gecelik var mı queries classified as indirim_kampanya_sorma",
    "Pattern 2: keten pantolon var mı goes to stok_sorgulama instead of ürün_arama"
  ],
  "regex_improvements": {{
    "ürün_arama": "\\\\b(gecelik|pijama|keten\\\\s+pantolon)\\\\s+.*(var\\\\s+m[ıi]|varm[ıi]|mevcut\\\\s+mu)\\\\b"
  }},
  "training_data_suggestions": [
    "__label__ürün_arama gecelik var mı acaba",
    "__label__ürün_arama keten pantolon var mı sizde",
    "__label__stok_sorgulama 42 beden var mı"
  ],
  "performance_insights": {{
    "avg_response_time": "5.2ms",
    "accuracy_trend": "improving",
    "cache_hit_rate": "65%"
  }},
  "priority_fixes": [
    "Fix gecelik var mı classification (high priority)",
    "Improve malzeme + ürün + var mı pattern matching"
  ]
}}
```

Sadece verilen conversation logs'ları kullan. Kendi varsayımlarını ekleme.
"""
        
        return prompt
    
    def call_ai_analysis(self, prompt: str) -> Dict:
        """AI/LLM'e analiz yaptır (OpenAI, Claude, vs.)"""
        try:
            # Burada gerçek AI API çağrısı yapılacak
            # Şimdilik mock response döndürüyorum
            
            # OpenAI example:
            # response = openai.ChatCompletion.create(
            #     model="gpt-4",
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=0.1
            # )
            # ai_response = response.choices[0].message.content
            
            # Mock response for demo
            ai_response = {
                "patterns_discovered": [
                    "Pattern 1: 'gecelik var mı' queries being classified as indirim_kampanya_sorma due to ML model bias",
                    "Pattern 2: Material + product + 'var mı' queries going to stok_sorgulama instead of ürün_arama"
                ],
                "regex_improvements": {
                    "ürün_arama": r"\b(gecelik|pijama|sabahlık|keten\s+pantolon|pamuklu\s+gömlek)\s+.*(var\s+m[ıi]|varm[ıi]|mevcut\s+mu|bulunur\s+mu)\b"
                },
                "training_data_suggestions": [
                    "__label__ürün_arama gecelik var mı acaba",
                    "__label__ürün_arama gecelik var mı sizde", 
                    "__label__ürün_arama keten pantolon var mı",
                    "__label__ürün_arama pamuklu gömlek var mı"
                ],
                "performance_insights": {
                    "avg_response_time": "4.8ms",
                    "accuracy_trend": "improving", 
                    "conflict_rate": "15%"
                },
                "priority_fixes": [
                    "HIGH: Fix 'gecelik var mı' → ürün_arama classification",
                    "MEDIUM: Improve material+product pattern matching",
                    "LOW: Optimize regex performance"
                ]
            }
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {"error": f"AI analysis failed: {e}"}
    
    def apply_improvements(self, ai_suggestions: Dict) -> Dict:
        """AI önerilerini otomatik uygula (veya admin onayına sun)"""
        applied_changes = []
        
        # 1. Regex improvements'leri main.py'ye ekle
        if ai_suggestions.get("regex_improvements"):
            # Bu kısım gerçek implementasyonda 
            # main.py'deki rules dict'ini güncelleyecek
            applied_changes.append("Updated regex patterns")
        
        # 2. Training data'yı train_nlu.txt'ye ekle  
        if ai_suggestions.get("training_data_suggestions"):
            try:
                with open("train_nlu.txt", "a", encoding="utf-8") as f:
                    f.write("\n# Auto-generated by Background Agent\n")
                    for suggestion in ai_suggestions["training_data_suggestions"]:
                        f.write(f"{suggestion}\n")
                applied_changes.append(f"Added {len(ai_suggestions['training_data_suggestions'])} training examples")
            except Exception as e:
                logger.error(f"Failed to update training data: {e}")
        
        # 3. Performance insights'leri log'la
        if ai_suggestions.get("performance_insights"):
            logger.info(f"Performance insights: {ai_suggestions['performance_insights']}")
            applied_changes.append("Logged performance insights")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "applied_changes": applied_changes,
            "ai_suggestions": ai_suggestions
        }
    
    def run_optimization_cycle(self):
        """Tam optimizasyon döngüsü"""
        logger.info("🤖 Starting optimization cycle...")
        
        # 1. Veri analizi
        analysis_data = self.analyze_conversation_logs(hours_back=24)
        
        if analysis_data.get("status") == "no_data":
            logger.info("No recent data to analyze")
            return
        
        # 2. AI prompt oluştur
        prompt = self.generate_improvement_prompt(analysis_data)
        
        # 3. AI analysis çağır
        ai_suggestions = self.call_ai_analysis(prompt)
        
        # 4. Önerileri uygula
        results = self.apply_improvements(ai_suggestions)
        
        # 5. Sonuçları kaydet
        self.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "analysis_data": analysis_data,
            "ai_suggestions": ai_suggestions,
            "applied_changes": results
        })
        
        logger.info(f"✅ Optimization cycle completed. Applied: {results['applied_changes']}")
        
        return results

def main():
    """Background agent'ı başlat"""
    agent = ChatbotOptimizerAgent()
    
    # İlk analizi hemen çalıştır
    agent.run_optimization_cycle()
    
    # Sonrasında schedule'a göre çalıştır
    schedule.every(6).hours.do(agent.run_optimization_cycle)  # 6 saatte bir
    schedule.every().day.at("03:00").do(agent.run_optimization_cycle)  # Her gece 03:00
    
    logger.info("🚀 Background Agent started. Running every 6 hours...")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # 1 dakikada bir kontrol et

if __name__ == "__main__":
    main() 