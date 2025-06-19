#!/usr/bin/env python3
"""
Test için optimize edilmiş main.py
Lifespan'ı devre dışı bırakır
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import Optional
import logging

# Test ortamı ayarları
os.environ["TESTING"] = "true"
os.environ["FAST_MODE"] = "true"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).parent

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    tenant_id: int
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    detected_intent: str
    confidence: float
    bot_response: str
    session_id: str
    tenant_id: int

# Test app'i oluştur - LIFESPAN YOK!
app = FastAPI()

# CORS
from fastapi.middleware.cors import CORSMiddleware
origins = ["http://localhost", "http://localhost:8000", "null"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fast mode intent detection
def detect_intent_fast_mode(query: str) -> tuple[str, float]:
    """Test ortamı için hızlı intent detection"""
    query_lower = query.lower().strip()
    
    # Intent patterns
    patterns = {
        "selamlama": [
            r"\b(merhaba|selam|mrb|slm|hey|hi|hello)\b",
            r"\b(günaydın|iyi günler|iyi akşamlar)\b"
        ],
        "tesekkur": [
            r"\b(teşekkür|teşekkürler|sağol|sağolun|thanks|thank you)\b",
            r"\b(çok teşekkür|teşekkür ederim)\b"
        ],
        "fiyat_sorgulama": [
            r"\b(fiyat|fiyatı|kaç para|ne kadar|eder|tutar)\b",
            r"\b(kaç tl|kaç lira|ücret|bedel)\b"
        ],
        "stok_sorgulama": [
            r"\b(stok|stokta|var mı|mevcut|kaldı|tükendi)\b",
            r"\b(beden|numara|size)\s+(var|yok|kaldı|tükendi)\b"
        ],
        "kargo_bilgisi_sorma": [
            r"\b(kargo|teslimat|gönderim|shipping|delivery)\b",
            r"\b(kaç gün|ne zaman|süre|ücret)\b"
        ],
        "iade_sorgulama": [
            r"\b(iade|değişim|değiştirme|geri verme)\b",
            r"\b(koşul|süre|nasıl|yapabilir)\b"
        ],
        "adres_sorma": [
            r"\b(adres|konum|yer|mağaza|dükkan)\b",
            r"\b(nerede|hangi|sokak|cadde)\b"
        ],
        "telefon_sorma": [
            r"\b(telefon|numara|numarası|iletişim)\b",
            r"\b(ara|arayabilir|ulaşabilir)\b"
        ],
        "odeme_yontemleri_sorma": [
            r"\b(ödeme|taksit|kredi kartı|havale|eft)\b",
            r"\b(kapıda ödeme|online|banka)\b"
        ],
        "siparis_durumu_sorma": [
            r"\b(sipariş|siparişim|durum|takip)\b",
            r"\b(nerede|ne zaman|geldi|gelmedi)\b"
        ],
        "urun_bilgisi_sorma": [
            r"\b(ürün|model|item|product)\b",
            r"\b(bilgi|detay|özellik|açıklama)\b"
        ],
        "tamam": [
            r"\b(tamam|ok|anladım|tamamladım)\b",
            r"\b(evet|hayır|tamam|peki)\b"
        ]
    }
    
    # Pattern matching
    import re
    for intent, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, query_lower):
                confidence = 0.9 if len(pattern_list) == 1 else 0.8
                return intent, confidence
    
    return "bilinmiyor", 0.1

def generate_response_fast_mode(intent: str, query: str, tenant_id: int) -> str:
    """Test ortamı için hızlı response generation"""
    
    responses = {
        "selamlama": "Merhaba! Size nasıl yardımcı olabilirim?",
        "tesekkur": "Rica ederim! Başka bir sorunuz var mı?",
        "fiyat_sorgulama": "Ürünün fiyatı 150 TL'dir.",
        "stok_sorgulama": "Evet, ürün stokta mevcut.",
        "kargo_bilgisi_sorma": "Kargo 2-3 gün içinde teslim edilir.",
        "iade_sorgulama": "14 gün içinde iade yapabilirsiniz.",
        "adres_sorma": "Mağazamız İstanbul, Kadıköy'de bulunmaktadır.",
        "telefon_sorma": "Telefon numaramız: 0212 345 67 89",
        "odeme_yontemleri_sorma": "Kredi kartı, havale ve kapıda ödeme seçeneklerimiz var.",
        "siparis_durumu_sorma": "Siparişiniz hazırlanıyor.",
        "urun_bilgisi_sorma": "Ürün hakkında detaylı bilgi verebilirim.",
        "tamam": "Tamam, başka bir sorunuz var mı?",
        "bilinmiyor": "Anlayamadım, lütfen tekrar sorar mısınız?"
    }
    
    return responses.get(intent, "Üzgünüm, bu konuda yardımcı olamıyorum.")

@app.post("/process_query/", response_model=QueryResponse)
async def process_query_test(payload: QueryRequest):
    """Test ortamı için optimize edilmiş query processing"""
    
    # Fast mode intent detection
    intent, confidence = detect_intent_fast_mode(payload.query)
    
    # Generate response
    response = generate_response_fast_mode(intent, payload.query, payload.tenant_id)
    
    # Generate session ID if not provided
    session_id = payload.session_id or f"test_session_{payload.tenant_id}"
    
    return QueryResponse(
        detected_intent=intent,
        confidence=confidence,
        bot_response=response,
        session_id=session_id,
        tenant_id=payload.tenant_id
    )

@app.get("/")
async def read_root():
    """Test endpoint"""
    return {
        "message": "Test Chatbot NLU API - Test Modu Aktif (Lifespan Yok)",
        "status": "running",
        "mode": "test",
        "lifespan": "disabled"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 