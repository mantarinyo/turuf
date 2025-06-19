#!/usr/bin/env python3
"""
Basit Test Server - Sadece temel intent detection
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import re

app = FastAPI(title="Basit Test Server", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    tenant_id: int
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    detected_intent: str
    response: str
    confidence: float

def detect_intent_simple(query: str) -> tuple[str, float]:
    """Basit intent detection - regex tabanlı"""
    query_lower = query.lower().strip()
    
    # Müşteri hizmetleri - en yüksek öncelik
    if any(word in query_lower for word in ["müşteri temsilcisi", "yetkili", "bağlar", "eksik geldi", "şikayet"]):
        return "musteri_hizmetlerine_baglanma", 0.9
    
    # İade sorgulama - yüksek öncelik
    if any(word in query_lower for word in ["iade", "değişim", "geri", "iptal", "14 gün"]):
        return "iade_sorgulama", 0.9
    
    # Kargo bilgisi - yüksek öncelik
    if any(word in query_lower for word in ["kargo", "teslimat", "gönderim", "ulaşır", "kac gun"]):
        return "kargo_bilgisi_sorma", 0.9
    
    # Ödeme yöntemleri - orta öncelik
    if any(word in query_lower for word in ["ödeme", "kapıda", "kredi kartı", "havale", "sipariş ver", "nasıl sipariş"]):
        return "odeme_yontemleri_sorma", 0.9
    
    # Lokasyon sorgulama - orta öncelik
    if any(word in query_lower for word in ["adres", "mağaza", "nerede", "lokasyon"]):
        return "lokasyon_sorma", 0.9
    
    # Telefon sorgulama - orta öncelik
    if any(word in query_lower for word in ["telefon", "numara", "ara"]):
        return "tel_no_sorma", 0.9
    
    # Çalışma saatleri - orta öncelik
    if any(word in query_lower for word in ["çalışma", "saat", "açık", "kapalı"]):
        return "calisma_saatleri_sorma", 0.9
    
    # Stok sorgulama - düşük öncelik (sadece beden/renk ile)
    if any(word in query_lower for word in ["beden", "renk", "stok"]) and any(word in query_lower for word in ["var mı", "kaldı mı", "mevcut"]):
        return "stok_sorgulama", 0.9
    
    # Fiyat sorgulama - düşük öncelik
    if any(word in query_lower for word in ["fiyat", "ne kadar", "kaç para", "ücret"]) and "kargo" not in query_lower:
        return "fiyat_sorgulama", 0.9
    
    # Ürün malzeme sorgulama
    if any(word in query_lower for word in ["kumaş", "malzeme", "pamuk", "polyester"]):
        return "ürün_malzeme_sorma", 0.9
    
    # Ürün bilgisi sorgulama
    if any(word in query_lower for word in ["renk", "model", "marka", "orijinal"]) and "var mı" in query_lower:
        return "ürün_bilgisi_sorma", 0.8
    
    # Selamlama
    if any(word in query_lower for word in ["merhaba", "selam", "mrb", "slm"]):
        return "selamlama", 0.9
    
    # Teşekkür
    if any(word in query_lower for word in ["teşekkür", "sağol", "teşekkürler"]):
        return "tesekkur", 0.9
    
    # Olumsuz yanıt
    if any(word in query_lower for word in ["hayır", "yok", "hayır", "olmaz"]):
        return "olumsuz_yanıt", 0.9
    
    # Bilinmiyor
    return "bilinmiyor", 0.5

def generate_response(intent: str, query: str) -> str:
    """Basit response generation"""
    responses = {
        "stok_sorgulama": "Ürün stok durumunu kontrol ediyorum...",
        "fiyat_sorgulama": "Ürün fiyatını öğreniyorum...",
        "kargo_bilgisi_sorma": "Kargo bilgilerini veriyorum...",
        "iade_sorgulama": "İade koşullarını açıklıyorum...",
        "selamlama": "Merhaba! Size nasıl yardımcı olabilirim?",
        "tesekkur": "Rica ederim! Başka bir konuda yardımcı olabilir miyim?",
        "ürün_bilgisi_sorma": "Ürün hakkında bilgi veriyorum...",
        "odeme_yontemleri_sorma": "Ödeme yöntemlerini açıklıyorum...",
        "bilinmiyor": "Ne demek istediğinizi tam anlayamadım. Lütfen farklı bir şekilde sorabilir misiniz?"
    }
    return responses.get(intent, "Üzgünüm, bu konuda yardımcı olamıyorum.")

@app.post("/process_query/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Ana endpoint"""
    intent, confidence = detect_intent_simple(request.query)
    response = generate_response(intent, request.query)
    
    return QueryResponse(
        detected_intent=intent,
        response=response,
        confidence=confidence
    )

@app.get("/")
async def root():
    return {"message": "Basit Test Server Çalışıyor!", "status": "active"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080) 