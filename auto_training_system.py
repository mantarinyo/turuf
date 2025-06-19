#!/usr/bin/env python3
"""
Otomatik Training Data Üretme Sistemi
- Yeni ürünlerden training data üret
- Kullanıcı etkileşimlerinden öğren
- Hatalı tahminleri düzelt
"""

import sqlite3
import json
import re
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AutoTrainingSystem:
    def __init__(self, db_path: str = "chatbot_data.db"):
        self.db_path = db_path
        
    def generate_product_training_data(self, product_data: Dict[str, Any]) -> List[str]:
        """Yeni ürün verilerinden training data üret"""
        
        product_name = product_data.get('name', '').lower()
        category = product_data.get('category', '').lower()
        brand = product_data.get('brand', '').lower()
        price = product_data.get('price', 0)
        
        training_samples = []
        
        # Ürün arama örnekleri
        product_search_templates = [
            f"__label__ürün_arama {product_name} var mı",
            f"__label__ürün_arama {product_name} mevcut mu",
            f"__label__ürün_arama {product_name} stokta var mı",
            f"__label__ürün_arama {product_name} bulunur mu",
            f"__label__ürün_arama {product_name} satıyor musunuz",
            f"__label__ürün_arama {product_name} modelleriniz var mı",
            f"__label__ürün_arama {category} kategorisinde {product_name}",
            f"__label__ürün_arama {brand} marka {product_name}",
            f"__label__ürün_arama elinizde {product_name} var mı",
            f"__label__ürün_arama {product_name} arıyorum",
        ]
        
        # Fiyat sorgulama örnekleri
        price_templates = [
            f"__label__fiyat_sorgulama {product_name} fiyatı nedir",
            f"__label__fiyat_sorgulama {product_name} ne kadar",
            f"__label__fiyat_sorgulama {product_name} kaça",
            f"__label__fiyat_sorgulama {product_name} fiyat bilgisi",
            f"__label__fiyat_sorgulama {product_name} ücreti",
            f"__label__fiyat_sorgulama {brand} {product_name} fiyatı",
        ]
        
        # Stok sorgulama örnekleri
        stock_templates = [
            f"__label__stok_sorgulama {product_name} stokta mı",
            f"__label__stok_sorgulama {product_name} var mı elinizde",
            f"__label__stok_sorgulama {product_name} kaldı mı",
            f"__label__stok_sorgulama {product_name} tükenmiş mi",
            f"__label__stok_sorgulama {product_name} hangi bedenler var",
        ]
        
        training_samples.extend(product_search_templates)
        training_samples.extend(price_templates)
        training_samples.extend(stock_templates)
        
        return training_samples
    
    def learn_from_user_interactions(self, days: int = 7) -> List[str]:
        """Son N günün kullanıcı etkileşimlerinden öğren"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Yanlış tahmin edilen sorguları bul
            cursor.execute('''
                SELECT query, detected_intent, user_feedback, correction
                FROM conversation_logs 
                WHERE created_at >= date('now', '-{} days')
                AND (user_feedback = 'negative' OR correction IS NOT NULL)
            '''.format(days))
            
            rows = cursor.fetchall()
            conn.close()
            
            corrected_samples = []
            
            for query, wrong_intent, feedback, correction in rows:
                if correction:
                    # Kullanıcı düzeltme yapmış
                    corrected_samples.append(f"__label__{correction} {query}")
                    logger.info(f"Öğrenildi: '{query}' -> {correction} (eski: {wrong_intent})")
            
            return corrected_samples
            
        except Exception as e:
            logger.error(f"User interaction learning error: {e}")
            return []
    
    def detect_pattern_gaps(self) -> List[str]:
        """Eksik pattern'leri tespit et ve training data üret"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Düşük confidence'lı tahminleri bul
            cursor.execute('''
                SELECT query, detected_intent, confidence_score
                FROM conversation_logs 
                WHERE confidence_score < 0.6 
                AND detected_intent != 'kapsam_disi'
                AND created_at >= date('now', '-30 days')
                GROUP BY query
                HAVING COUNT(*) > 1
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            pattern_samples = []
            
            for query, intent, confidence in rows:
                # Benzer pattern'ler üret
                variations = self.generate_query_variations(query)
                for variation in variations:
                    pattern_samples.append(f"__label__{intent} {variation}")
            
            return pattern_samples
            
        except Exception as e:
            logger.error(f"Pattern gap detection error: {e}")
            return []
    
    def generate_query_variations(self, original_query: str) -> List[str]:
        """Sorgu varyasyonları üret (basit augmentation)"""
        
        variations = []
        
        # Kelime değişimleri
        replacements = {
            'var mı': ['mevcut mu', 'bulunur mu', 'satıyor musunuz'],
            'fiyat': ['ücret', 'maliyet', 'para'],
            'ne kadar': ['kaça', 'kaç para', 'kaç lira'],
            'stok': ['elimizde', 'mevcut', 'satışta'],
        }
        
        for old_word, new_words in replacements.items():
            if old_word in original_query.lower():
                for new_word in new_words:
                    variation = original_query.lower().replace(old_word, new_word)
                    variations.append(variation)
        
        # Ek kelimeler
        prefixes = ['acaba', 'şey', 'bi']
        suffixes = ['acaba', 'ya', 'bakalım']
        
        for prefix in prefixes:
            variations.append(f"{prefix} {original_query}")
        
        for suffix in suffixes:
            variations.append(f"{original_query} {suffix}")
        
        return variations[:5]  # Maksimum 5 varyasyon
    
    def auto_retrain_model(self) -> bool:
        """Otomatik model yeniden eğitimi"""
        
        try:
            # Yeni training data'ları topla
            new_samples = []
            
            # 1. Yeni ürünlerden
            # Bu kısım product_importer.py ile entegre edilmeli
            
            # 2. Kullanıcı etkileşimlerinden
            interaction_samples = self.learn_from_user_interactions()
            new_samples.extend(interaction_samples)
            
            # 3. Pattern gap'lerden
            pattern_samples = self.detect_pattern_gaps()
            new_samples.extend(pattern_samples)
            
            if len(new_samples) < 10:
                logger.info("Yeterli yeni sample yok, retrain atlanıyor")
                return False
            
            # Mevcut train_nlu.txt'ye ekle
            with open('train_nlu.txt', 'a', encoding='utf-8') as f:
                f.write('\n')
                for sample in new_samples:
                    f.write(f"{sample}\n")
            
            logger.info(f"{len(new_samples)} yeni sample eklendi, model retrain başlıyor...")
            
            # FastText ile yeniden eğit
            import fasttext
            model = fasttext.train_supervised(
                input='train_nlu.txt',
                epoch=25,
                lr=0.5,
                wordNgrams=2,
                dim=100
            )
            
            # Yeni modeli kaydet
            model.save_model('nlu_model_updated.bin')
            
            # Eski modeli yedekle ve yenisini aktif et
            import shutil
            shutil.move('nlu_model.bin', f'nlu_model_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.bin')
            shutil.move('nlu_model_updated.bin', 'nlu_model.bin')
            
            logger.info("✅ Otomatik retrain tamamlandı!")
            return True
            
        except Exception as e:
            logger.error(f"Auto retrain error: {e}")
            return False

# Scheduled task için
def run_auto_training():
    """Günlük otomatik eğitim görevi"""
    trainer = AutoTrainingSystem()
    
    # Haftalık bir kez çalıştır
    today = datetime.now().weekday()
    if today == 6:  # Pazar günü
        trainer.auto_retrain_model()

if __name__ == "__main__":
    # Test
    trainer = AutoTrainingSystem()
    
    # Örnek yeni ürün
    new_product = {
        'name': 'Kışlık Trençkot',
        'category': 'Dış Giyim', 
        'brand': 'Zara',
        'price': 299.99
    }
    
    samples = trainer.generate_product_training_data(new_product)
    print("Üretilen training samples:")
    for sample in samples[:5]:
        print(sample) 