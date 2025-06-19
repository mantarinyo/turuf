#!/usr/bin/env python3
"""
Hafif NLU Model Oluşturucu
Büyük modeli küçültür ve optimize eder
"""
import fasttext
import os
from pathlib import Path

def create_lightweight_model():
    """Büyük modeli küçültür"""
    
    input_model = "nlu_model.bin"
    output_model = "nlu_model_light.bin"
    
    if not Path(input_model).exists():
        print(f"❌ {input_model} bulunamadı!")
        return False
    
    print(f"📦 Büyük model yükleniyor: {input_model}")
    
    try:
        # Büyük modeli yükle
        model = fasttext.load_model(input_model)
        
        print(f"✅ Model yüklendi. Boyut: {os.path.getsize(input_model) / (1024*1024):.1f} MB")
        
        # Modeli optimize et ve kaydet
        print("🔧 Model optimize ediliyor...")
        
        # Daha küçük model oluştur - quantize edilmiş
        model.quantize(input=output_model, qnorm=True, retrain=True, cutoff=100000)
        
        # Yeni model boyutunu kontrol et
        new_size = os.path.getsize(output_model) / (1024*1024)
        print(f"✅ Hafif model oluşturuldu: {output_model}")
        print(f"📊 Yeni boyut: {new_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

if __name__ == "__main__":
    create_lightweight_model() 