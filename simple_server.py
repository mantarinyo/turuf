#!/usr/bin/env python3
# simple_server.py - Gelişmiş chatbot sunucusu

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import sys
import os
import re
import json

sys.path.insert(0, os.path.abspath('.'))

from main import extract_simple_entities, _preprocess_text_for_matching

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    session_id: str = None
    tenant_id: int = 1

class QueryResponse(BaseModel):
    query: str
    detected_intent: str
    entities: dict
    bot_response: str
    confidence: float
    clarification_needed: bool = False

# Gelişmiş intent detection
def detect_intent_advanced(query: str) -> tuple[str, float]:
    """Gelişmiş intent tespiti"""
    query_lower = query.lower()
    
    # Intent patterns - daha kapsamlı
    intent_patterns = {
        "selamlama": [
            r"\b(merhaba|selam|slm|mrb|hey|günaydın|iyi günler|sa|selamun aleykum)\b"
        ],
        "fiyat_sorgulama": [
            r"\b(fiyat|ücret|kaç para|ne kadar|kaç tl|maliyet|ederi|kaça|nekadar|fyt|fiyt|fyaat|fiyay|kça para|ne kadr|fiyatı ne|fiyatı nedir|ücreti nedir|fiyatını öğren|fiyat bilgisi|kaç para|para)\b"
        ],
        "stok_sorgulama": [
            r"\b(stokta mevcut mu|stokta var mı|stok durumu|elde var mı|beden.*var mı|bednleri var mı|numarası var mı|modeli var mı|bulunur mu|kaldı mı|bedeni var mı|stok|bedenleri|mevcutmu|varmı|var mı)\b"
        ],
        "ürün_malzeme_sorma": [
            r"\b(malzeme|içeriğ.*|kumaş.*|astar|yapılmış|üretilmiş|neyden yapıl|materyal|kumas ne|kompozisyonu)\b"
        ],
        "ürün_bilgisi_sorma": [
            r"\b(özellik.*|hakkında bilgi|detay|açıklama|nedir bu|ne işe yarar|ürün bilgisi|ürünle ilgili|model hakkında|ürün ne için|anlatır mısın bu ürün|spesifikasyonları|kalıbı|dar mı|geniş mi|nasıl yıkanır)\b"
        ],
        "iade_sorgulama": [
            r"\b(iade|geri verme|değişim|değiştir|iade edebilir|iade koşul.*|koşullaeı|para iadesi|değiştirebilir miyim|geri gönderebilir|ürünü geri al|beğenmedim|nasıl yapılır)\b"
        ],
        "kargo_bilgisi_sorma": [
            r"\b(kargo|gönderim|teslimat|kaç günde gelir|kargo ücret|kargo ne kadar|kargo takip|yurtiçi kargo|kargo nekadar|kargonuz kaç günde|teslim süresi|kargo tutar|sipariş nasıl|ne kadar sürer|takip edebilir miyim)\b"
        ],
        "calisma_saatleri_sorma": [
            r"\b((?:çalışma|calisma) saatleri.*|kaça kadar açık|ne zaman açık|açılış kapanış|mesai|hafta sonu açık|pazar açık mı|hangi saatler|ne zaman kapanıyor|saat kaçta açılıyor|saat kaçta kapanır|açıksınız|calisma saati)\b"
        ],
        "lokasyon_sorma": [
            r"\b(nerede|adres|konum|yeriniz|mağaza nerede|dükkan nerede|nasıl gel|nerdesiniz|konm|adresiniz neydi|dükkan nerde|hangi semtte|yol tarifi|magazanız|maps|harita)\b"
        ],
        "tel_no_sorma": [
            r"\b(telefon|tel no|numara|iletişim no|arayabilir|whatsapp|mail|e-posta|eposta|numaranız|mail adresiniz|irtibat|nasıl ulaşabilirim)\b"
        ],
        "odeme_yontemleri_sorma": [
            r"\b(nasıl öde|ödeme seçenek|ne kabul|kredi kartı|taksit|kapıda ödeme|havale|eft|ödeme türleri|ödeme yapabilir|taksit imkanı|ödeme şekilleri|kabul ediyorsunuz|yapıyor musunuz)\b"
        ],
        "tesekkur": [
            r"\b(teşekkür|sağ ol|tşk|eyvallah|saol|mersi|eyw|tskler|tamam|tmm|ok|anladım|pekala|tamamdır|varol)\b"
        ],
        "oneri_isteme": [
            r"\b(ne önerirsin|tavsiye eder|en çok satan|benzer ne var|alternatif ne|öneri var mı|ne tavsiye|bir şey öner|hangi ürünü almalı|ne seçmeli)\b"
        ],
        "olumsuz_yanıt": [
            r"\b(hayır|yok kalsın|gerek yok|istemiyorum|düşünmüyorum|pas|vazgeçtim|kalsın|olmaz|hayr|ilgilenmiyorum|almayayım)\b"
        ]
    }
    
    # Her intent için confidence hesapla
    intent_scores = {}
    
    for intent, patterns in intent_patterns.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, query_lower):
                score += 1
        if score > 0:
            intent_scores[intent] = score
    
    # Çoklu intent tespiti - öncelik sırası
    priority_intents = [
        "selamlama", "tesekkur", "olumsuz_yanıt",  # En yüksek öncelik
        "iade_sorgulama", "kargo_bilgisi_sorma", "odeme_yontemleri_sorma",  # Orta öncelik
        "stok_sorgulama", "fiyat_sorgulama", "ürün_malzeme_sorma", "ürün_bilgisi_sorma",  # Düşük öncelik
        "lokasyon_sorma", "tel_no_sorma", "calisma_saatleri_sorma", "oneri_isteme"  # En düşük öncelik
    ]
    
    # Öncelik sırasına göre intent seç
    for priority_intent in priority_intents:
        if priority_intent in intent_scores:
            confidence = min(intent_scores[priority_intent] / 2.0, 1.0)
            return priority_intent, confidence
    
    # Eğer öncelikli intent yoksa, en yüksek skorlu olanı seç
    if intent_scores:
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(intent_scores[best_intent] / 2.0, 1.0)
        return best_intent, confidence
    
    return "bilinmiyor", 0.0

# Gelişmiş response generation
def generate_response(intent: str, entities: dict, query: str) -> tuple[str, bool]:
    """Gelişmiş cevap üretimi"""
    
    item_name = entities.get("item_name_candidate", "")
    size = entities.get("size", "")
    
    responses = {
        "selamlama": "Merhaba! Size nasıl yardımcı olabilirim?",
        "fiyat_sorgulama": f"Ürün fiyatları hakkında bilgi veriyorum{f' - {item_name}' if item_name else ''}.",
        "stok_sorgulama": f"Stok durumu hakkında bilgi veriyorum{f' - {item_name}' if item_name else ''}{f' ({size} beden)' if size else ''}.",
        "ürün_malzeme_sorma": f"Ürün malzemesi hakkında bilgi veriyorum{f' - {item_name}' if item_name else ''}.",
        "ürün_bilgisi_sorma": f"Ürün özellikleri hakkında bilgi veriyorum{f' - {item_name}' if item_name else ''}.",
        "iade_sorgulama": "İade ve değişim koşulları hakkında bilgi veriyorum.",
        "kargo_bilgisi_sorma": "Kargo ve teslimat bilgileri hakkında bilgi veriyorum.",
        "calisma_saatleri_sorma": "Çalışma saatleri hakkında bilgi veriyorum.",
        "lokasyon_sorma": "Mağaza adresi ve konum bilgileri hakkında bilgi veriyorum.",
        "tel_no_sorma": "İletişim bilgileri hakkında bilgi veriyorum.",
        "odeme_yontemleri_sorma": "Ödeme yöntemleri hakkında bilgi veriyorum.",
        "tesekkur": "Rica ederim! Başka bir konuda yardımcı olabilir miyim?",
        "oneri_isteme": "Size en uygun ürünleri öneriyorum.",
        "olumsuz_yanıt": "Anladım. Başka bir konuda yardımcı olabilir miyim?",
        "bilinmiyor": "Ne demek istediğinizi tam anlayamadım. Lütfen farklı bir şekilde sorabilir misiniz?"
    }
    
    response = responses.get(intent, responses["bilinmiyor"])
    clarification_needed = False
    
    # Netleştirme gereken durumlar
    if intent == "bilinmiyor" and len(query.split()) < 3:
        clarification_needed = True
        response = "Hangi ürün hakkında bilgi almak istiyorsunuz?"
    
    return response, clarification_needed

@app.get("/")
async def root():
    return {"message": "Simple Test Server Çalışıyor!", "status": "active"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Server çalışıyor"}

@app.post("/process_query/")
async def process_query(request: QueryRequest):
    return {
        "original_query": request.query,
        "session_id": request.session_id or "test_session",
        "tenant_id": request.tenant_id,
        "nlu_method": "simple_test",
        "detected_intent": "test_intent",
        "bot_response": f"Test cevabı: {request.query}",
        "ask_for_clarification": False
    }

if __name__ == "__main__":
    print("🚀 Basit Test Server Başlatılıyor...")
    print("🌐 Server: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info") 