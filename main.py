# main.py
import sys
print('--- main.py import başı ---')
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import re
import fasttext # type: ignore
import zeyrek # type: ignore
from rapidfuzz import process, fuzz
from pathlib import Path
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
import logging
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import sqlite3
import json
import threading
from symspellpy import SymSpell, Verbosity # type: ignore
from zeyrek import MorphAnalyzer
from collections import defaultdict
import time
import concurrent.futures
import signal

# Kendi database_service modülünüzü import edin
# import database_service  # Supabase kullanıyor, SQLite kullanacağız

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "nlu_model.bin"
TURKISH_FREQUENCY_DICTIONARY_PATH = BASE_DIR / "turkish_frequency_dictionary.txt"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# Hızlı mod ayarı - test sırasında ağır modelleri devre dışı bırakır
FAST_MODE = True  # True: Sadece regex, False: ML + regex hibrit

# Global değişkenler - Lazy loading için
current_nlu_model: Optional[fasttext.FastText._FastText] = None
current_morphology: Optional[zeyrek.MorphAnalyzer] = None
current_sym_spell: Optional[SymSpell] = None
models_loaded = False

# FastText kullanılabilir
FASTTEXT_AVAILABLE = True

# --- SQLite Database Service ---
class SQLiteDatabaseService:
    def __init__(self, db_path: str = "chatbot_data.db"):
        self.db_path = Path(db_path)
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Veritabanı tablolarını oluştur"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tenants tablosu - MVP billing plan ile
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tenants (
                    id INTEGER PRIMARY KEY,
                    business_name TEXT NOT NULL,
                    business_type TEXT NOT NULL,
                    settings_json TEXT,
                    billing_plan TEXT DEFAULT 'basic',
                    monthly_query_limit INTEGER DEFAULT 1000,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tenant usage tracking tablosu - MVP için kritik
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tenant_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id INTEGER NOT NULL,
                    date DATE NOT NULL,
                    query_count INTEGER DEFAULT 0,
                    energy_consumption REAL DEFAULT 0.0,
                    avg_response_time REAL DEFAULT 0.0,
                    error_count INTEGER DEFAULT 0,
                    ml_queries INTEGER DEFAULT 0,
                    regex_queries INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tenant_id) REFERENCES tenants (id),
                    UNIQUE(tenant_id, date)
                )
            ''')
            
            # Billing history tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS billing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id INTEGER NOT NULL,
                    billing_period TEXT NOT NULL,
                    total_queries INTEGER DEFAULT 0,
                    total_energy REAL DEFAULT 0.0,
                    base_cost REAL DEFAULT 0.0,
                    overage_cost REAL DEFAULT 0.0,
                    total_cost REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tenant_id) REFERENCES tenants (id)
                )
            ''')
            
            # Clothing products tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS clothing_products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    price REAL,
                    category TEXT,
                    attributes_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tenant_id) REFERENCES tenants (id)
                )
            ''')
            
            # Sessions tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    tenant_id INTEGER NOT NULL,
                    context_data TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tenant_id) REFERENCES tenants (id)
                )
            ''')
            
            # Conversation logs tablosu - energy tracking ile
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    user_query_original TEXT,
                    user_query_spell_checked TEXT,
                    user_query_lemmatized TEXT,
                    detected_intent TEXT,
                    nlu_method TEXT,
                    slm_intent TEXT,
                    slm_confidence REAL,
                    entities_extracted TEXT,
                    resolved_item_id INTEGER,
                    resolved_item_name TEXT,
                    bot_response TEXT,
                    ask_for_clarification BOOLEAN,
                    clarification_options_offered TEXT,
                    response_time_ms REAL,
                    energy_consumption REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (tenant_id) REFERENCES tenants (id)
                )
            ''')
            
            # İndeksler
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_products_tenant ON clothing_products(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_products_name ON clothing_products(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_tenant ON conversation_logs(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_session ON conversation_logs(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_tenant_date ON tenant_usage(tenant_id, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_billing_tenant ON billing_history(tenant_id)')
            
            conn.commit()
            conn.close()
            
            # Örnek veri ekle (test için) - GÜVENLİ VERSİYON
            success = self._insert_sample_data_safe()
            if not success:
                logger.warning('LIFESPAN: Örnek veri ekleme başarısız, devam ediliyor')
                print('LIFESPAN: Örnek veri ekleme başarısız, devam ediliyor')
    
    def _insert_sample_data_safe(self):
        """Güvenli örnek veri ekleme - timeout ve hata toleranslı"""
        def _insert_data():
            try:
                logger.info('LIFESPAN: Örnek veri ekleme başladı')
                print('LIFESPAN: Örnek veri ekleme başladı')
                
                # Dosya izinlerini kontrol et
                if not os.access(self.db_path, os.W_OK):
                    logger.error('LIFESPAN: Database dosyası yazılabilir değil')
                    print('LIFESPAN: Database dosyası yazılabilir değil')
                    return False
                
                # Bağlantı timeout'u ile
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
                conn.execute("PRAGMA synchronous=NORMAL")  # Daha hızlı
                
                cursor = conn.cursor()
                
                # Transaction başlat
                cursor.execute("BEGIN TRANSACTION")
                
                try:
                    # Tenant kontrolü
                    cursor.execute('SELECT COUNT(*) FROM tenants WHERE id = 1')
                    if cursor.fetchone()[0] == 0:
                        cursor.execute('''
                            INSERT INTO tenants (id, business_name, business_type, settings_json) 
                            VALUES (1, 'Mantarinyo Giyim', 'giyim', '{"default_responses": {"greeting": "Merhaba! Size nasıl yardımcı olabilirim?"}}')
                        ''')
                        logger.info('LIFESPAN: Tenant eklendi')
                        print('LIFESPAN: Tenant eklendi')
                    
                    # Ürün kontrolü
                    cursor.execute('SELECT COUNT(*) FROM clothing_products WHERE tenant_id = 1')
                    if cursor.fetchone()[0] == 0:
                        # Sadece 3 temel ürün ekle (hızlı test için)
                        products = [
                            (1, 'Keten Pantolon', 299.99, 'Pantolon', '{"malzeme": "Keten", "renkler": ["Bej", "Kahverengi"], "bedenler": ["S", "M", "L"]}'),
                            (1, 'Pamuklu Gömlek', 199.99, 'Gömlek', '{"malzeme": "Pamuk", "renkler": ["Beyaz", "Mavi"], "bedenler": ["S", "M", "L"]}'),
                            (1, 'Kadın Elbise', 399.99, 'Elbise', '{"malzeme": "Polyester", "renkler": ["Siyah", "Kırmızı"], "bedenler": ["S", "M", "L"]}')
                        ]
                        
                        for product in products:
                            cursor.execute('''
                                INSERT INTO clothing_products (tenant_id, name, price, category, attributes_json)
                                VALUES (?, ?, ?, ?, ?)
                            ''', product)
                        
                        logger.info('LIFESPAN: Ürünler eklendi')
                        print('LIFESPAN: Ürünler eklendi')
                    
                    # Transaction commit
                    cursor.execute("COMMIT")
                    conn.close()
                    
                    logger.info('LIFESPAN: Örnek veri ekleme başarılı')
                    print('LIFESPAN: Örnek veri ekleme başarılı')
                    return True
                    
                except Exception as e:
                    # Rollback on error
                    cursor.execute("ROLLBACK")
                    conn.close()
                    raise e
                    
            except Exception as e:
                logger.error(f'LIFESPAN: Örnek veri ekleme HATA: {e}')
                print(f'LIFESPAN: Örnek veri ekleme HATA: {e}')
                return False
        
        # Timeout ile çalıştır
        result, error = run_with_timeout(_insert_data, timeout_seconds=15)
        
        if error == "timeout":
            logger.error('LIFESPAN: Örnek veri ekleme timeout')
            print('LIFESPAN: Örnek veri ekleme timeout')
            return False
        elif error:
            logger.error(f'LIFESPAN: Örnek veri ekleme thread hatası: {error}')
            print(f'LIFESPAN: Örnek veri ekleme thread hatası: {error}')
            return False
        
        return result
    
    def _insert_sample_data(self):
        """Eski örnek veri ekleme fonksiyonu - geriye uyumluluk için"""
        return self._insert_sample_data_safe()
    
    async def get_tenant_settings(self, tenant_id: int) -> Optional[dict]:
        """Tenant ayarlarını getir"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, business_name, business_type, settings_json
                    FROM tenants WHERE id = ?
                ''', (tenant_id,))
                
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return {
                        "id": row[0],
                        "business_name": row[1],
                        "business_type": row[2],
                        "settings_json": json.loads(row[3]) if row[3] else {}
                    }
                return None
        except Exception as e:
            logger.error(f"Tenant {tenant_id} ayarları çekilirken hata: {e}", exc_info=True)
            return None
    
    async def get_items_by_name_fuzzy(self, tenant_id: int, business_type: str, item_name_candidate: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fuzzy arama ile ürün bul"""
        if business_type.strip().lower() != "giyim":
            return []
        
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                if not item_name_candidate or not item_name_candidate.strip():
                    # Boş arama - limit kadar ürün getir
                    cursor.execute('''
                        SELECT id, name, price, category, attributes_json
                        FROM clothing_products 
                        WHERE tenant_id = ?
                        LIMIT ?
                    ''', (tenant_id, limit))
                else:
                    # Tüm ürünleri getir ve fuzzy arama yap
                    cursor.execute('''
                        SELECT id, name, price, category, attributes_json
                        FROM clothing_products 
                        WHERE tenant_id = ?
                    ''', (tenant_id,))
                
                rows = cursor.fetchall()
                conn.close()
                
                if not rows:
                    return []
                
                # Fuzzy arama
                products = []
                for row in rows:
                    products.append({
                        "id": row[0],
                        "name": row[1],
                        "price": row[2],
                        "category": row[3],
                        "attributes_json": json.loads(row[4]) if row[4] else {}
                    })
                
                if not item_name_candidate or not item_name_candidate.strip():
                    return products[:limit]
                
                # Fuzzy matching
                product_names = [p['name'] for p in products]
                product_map = {p['name']: p for p in products}
                
                best_matches = process.extract(
                    item_name_candidate, 
                    product_names, 
                    scorer=fuzz.WRatio, 
                    limit=limit, 
                    score_cutoff=60
                )
                
                return [product_map[match[0]] for match in best_matches]
                
        except Exception as e:
            logger.error(f"Fuzzy arama hatası: {e}", exc_info=True)
            return []
    
    async def get_item_by_id(self, tenant_id: int, business_type: str, item_id: Any) -> Optional[Dict[str, Any]]:
        """ID ile ürün getir"""
        if business_type.strip().lower() != "giyim":
            return None
        
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, name, price, category, attributes_json
                    FROM clothing_products 
                    WHERE tenant_id = ? AND id = ?
                ''', (tenant_id, item_id))
                
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return {
                        "id": row[0],
                        "name": row[1],
                        "price": row[2],
                        "category": row[3],
                        "attributes_json": json.loads(row[4]) if row[4] else {}
                    }
                return None
        except Exception as e:
            logger.error(f"Ürün getirme hatası: {e}", exc_info=True)
            return None
    
    async def get_session_data(self, session_id: str, tenant_id: int) -> Optional[dict]:
        """Session verilerini getir"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT context_data FROM sessions 
                    WHERE session_id = ? AND tenant_id = ?
                ''', (session_id, tenant_id))
                
                row = cursor.fetchone()
                conn.close()
                
                if row and row[0]:
                    return json.loads(row[0])
                return {}
        except Exception as e:
            logger.error(f"Session veri getirme hatası: {e}", exc_info=True)
            return {}
    
    async def save_session_data(self, session_id: str, tenant_id: int, data: dict) -> bool:
        """Session verilerini kaydet"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO sessions (session_id, tenant_id, context_data, last_updated)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, tenant_id, json.dumps(data), datetime.now(timezone.utc).isoformat()))
                
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"Session kaydetme hatası: {e}", exc_info=True)
            return False
    
    async def log_conversation_turn(self, log_data: dict) -> bool:
        """Konuşma turunu logla"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO conversation_logs (
                        tenant_id, session_id, user_query_original, user_query_spell_checked,
                        user_query_lemmatized, detected_intent, nlu_method, slm_intent,
                        slm_confidence, entities_extracted, resolved_item_id, resolved_item_name,
                        bot_response, ask_for_clarification, clarification_options_offered,
                        response_time_ms, energy_consumption, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    log_data.get('tenant_id'), log_data.get('session_id'),
                    log_data.get('user_query_original'), log_data.get('user_query_spell_checked'),
                    log_data.get('user_query_lemmatized'), log_data.get('detected_intent'),
                    log_data.get('nlu_method'), log_data.get('slm_intent'),
                    log_data.get('slm_confidence'), log_data.get('entities_extracted'),
                    log_data.get('resolved_item_id'), log_data.get('resolved_item_name'),
                    log_data.get('bot_response'), log_data.get('ask_for_clarification'),
                    log_data.get('clarification_options_offered'), log_data.get('response_time_ms'),
                    log_data.get('energy_consumption', 0.0), datetime.now()
                ))
                
                conn.commit()
                conn.close()
                return True
                
        except Exception as e:
            logger.error(f'Log conversation error: {e}')
            return False
    
    # MVP Tenant Usage Tracking Methods
    async def update_tenant_usage(self, tenant_id: int, response_time: float, energy_consumption: float, 
                                nlu_method: str, success: bool = True) -> bool:
        """Tenant kullanım istatistiklerini güncelle"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                today = datetime.now().date().isoformat()
                
                # Mevcut günlük kayıt var mı kontrol et
                cursor.execute('''
                    SELECT id, query_count, energy_consumption, avg_response_time, error_count, ml_queries, regex_queries
                    FROM tenant_usage WHERE tenant_id = ? AND date = ?
                ''', (tenant_id, today))
                
                row = cursor.fetchone()
                
                if row:
                    # Mevcut kaydı güncelle
                    record_id, current_count, current_energy, current_avg_time, current_errors, current_ml, current_regex = row
                    
                    new_count = current_count + 1
                    new_energy = current_energy + energy_consumption
                    new_avg_time = ((current_avg_time * current_count) + response_time) / new_count
                    new_errors = current_errors + (0 if success else 1)
                    new_ml = current_ml + (1 if 'ml' in nlu_method.lower() else 0)
                    new_regex = current_regex + (1 if 'regex' in nlu_method.lower() else 0)
                    
                    cursor.execute('''
                        UPDATE tenant_usage SET 
                            query_count = ?, energy_consumption = ?, avg_response_time = ?,
                            error_count = ?, ml_queries = ?, regex_queries = ?
                        WHERE id = ?
                    ''', (new_count, new_energy, new_avg_time, new_errors, new_ml, new_regex, record_id))
                    
                else:
                    # Yeni günlük kayıt oluştur
                    cursor.execute('''
                        INSERT INTO tenant_usage (
                            tenant_id, date, query_count, energy_consumption, avg_response_time,
                            error_count, ml_queries, regex_queries
                        ) VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                    ''', (
                        tenant_id, today, energy_consumption, response_time,
                        0 if success else 1,
                        1 if 'ml' in nlu_method.lower() else 0,
                        1 if 'regex' in nlu_method.lower() else 0
                    ))
                
                conn.commit()
                conn.close()
                return True
                
        except Exception as e:
            logger.error(f'Update tenant usage error: {e}')
            return False
    
    async def get_tenant_usage(self, tenant_id: int, days: int = 30) -> Dict[str, Any]:
        """Tenant kullanım istatistiklerini getir"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Son N günün verilerini al
                cursor.execute('''
                    SELECT date, query_count, energy_consumption, avg_response_time, 
                           error_count, ml_queries, regex_queries
                    FROM tenant_usage 
                    WHERE tenant_id = ? AND date >= date('now', '-{} days')
                    ORDER BY date DESC
                '''.format(days), (tenant_id,))
                
                rows = cursor.fetchall()
                
                # Toplam istatistikler
                total_queries = sum(row[1] for row in rows)
                total_energy = sum(row[2] for row in rows)
                total_errors = sum(row[4] for row in rows)
                total_ml = sum(row[5] for row in rows)
                total_regex = sum(row[6] for row in rows)
                
                avg_response_time = sum(row[3] for row in rows) / len(rows) if rows else 0
                
                # Tenant bilgilerini al
                cursor.execute('''
                    SELECT billing_plan, monthly_query_limit FROM tenants WHERE id = ?
                ''', (tenant_id,))
                
                tenant_row = cursor.fetchone()
                billing_plan = tenant_row[0] if tenant_row else 'basic'
                monthly_limit = tenant_row[1] if tenant_row else 1000
                
                conn.close()
                
                return {
                    "tenant_id": tenant_id,
                    "billing_plan": billing_plan,
                    "monthly_limit": monthly_limit,
                    "total_queries": total_queries,
                    "total_energy": round(total_energy, 4),
                    "avg_response_time": round(avg_response_time * 1000, 2),  # ms
                    "error_rate": round((total_errors / total_queries * 100), 2) if total_queries > 0 else 0,
                    "ml_usage_rate": round((total_ml / total_queries * 100), 2) if total_queries > 0 else 0,
                    "regex_usage_rate": round((total_regex / total_queries * 100), 2) if total_queries > 0 else 0,
                    "usage_percentage": round((total_queries / monthly_limit * 100), 2),
                    "daily_data": [
                        {
                            "date": row[0],
                            "queries": row[1],
                            "energy": round(row[2], 4),
                            "avg_time": round(row[3] * 1000, 2),
                            "errors": row[4]
                        } for row in rows
                    ]
                }
                
        except Exception as e:
            logger.error(f'Get tenant usage error: {e}')
            return {}
    
    async def calculate_billing(self, tenant_id: int, month: str = None) -> Dict[str, Any]:
        """Tenant fatura hesaplama"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Tenant bilgilerini al
                cursor.execute('''
                    SELECT business_name, billing_plan, monthly_query_limit
                    FROM tenants WHERE id = ?
                ''', (tenant_id,))
                
                tenant_row = cursor.fetchone()
                if not tenant_row:
                    return None
                
                business_name, billing_plan, monthly_limit = tenant_row
                
                # Son 30 günlük kullanım
                cursor.execute('''
                    SELECT COUNT(*) as total_queries,
                           SUM(energy_consumption) as total_energy,
                           AVG(response_time_ms) as avg_response_time
                    FROM conversation_logs 
                    WHERE tenant_id = ? AND created_at >= date('now', '-30 days')
                ''', (tenant_id,))
                
                usage_row = cursor.fetchone()
                total_queries = usage_row[0] if usage_row else 0
                total_energy = usage_row[1] if usage_row and usage_row[1] else 0
                avg_response_time = usage_row[2] if usage_row and usage_row[2] else 0
                
                # Fiyatlandırma hesaplama
                base_cost = 50  # Temel aylık ücret
                overage_cost = 0
                
                if total_queries > monthly_limit:
                    overage_queries = total_queries - monthly_limit
                    overage_cost = overage_queries * 0.01  # Her fazla sorgu 1 kuruş
                
                total_cost = base_cost + overage_cost
                usage_percentage = (total_queries / monthly_limit * 100) if monthly_limit > 0 else 0
                
                conn.close()
                
                return {
                    "tenant_id": tenant_id,
                    "business_name": business_name,
                    "billing_plan": billing_plan,
                    "billing_period": "Son 30 gün",
                    "monthly_limit": monthly_limit,
                    "total_queries": total_queries,
                    "total_energy": round(total_energy, 4),
                    "avg_response_time": round(avg_response_time, 2),
                    "usage_percentage": round(usage_percentage, 2),
                    "base_cost": base_cost,
                    "overage_cost": round(overage_cost, 2),
                    "total_cost": round(total_cost, 2)
                }
                
        except Exception as e:
            logger.error(f"Calculate billing error: {e}")
            return None

    async def get_tenant_billing(self, tenant_id: int) -> dict:
        """Tenant fatura özeti"""
        return await self.calculate_billing(tenant_id)

    async def get_tenants_usage_summary(self) -> dict:
        """Tüm tenant'ların kullanım özeti"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT t.id, t.business_name, t.billing_plan, t.monthly_query_limit,
                           COALESCE(COUNT(cl.id), 0) as total_queries,
                           COALESCE(SUM(cl.energy_consumption), 0) as total_energy,
                           COALESCE(AVG(cl.response_time_ms), 0) as avg_response_time
                    FROM tenants t
                    LEFT JOIN conversation_logs cl ON t.id = cl.tenant_id 
                        AND cl.created_at >= date('now', '-30 days')
                    GROUP BY t.id, t.business_name, t.billing_plan, t.monthly_query_limit
                    ORDER BY total_queries DESC
                ''')
                
                rows = cursor.fetchall()
                conn.close()
                
                tenants_summary = []
                total_queries = 0
                total_energy = 0.0
                
                for row in rows:
                    tenant_id, business_name, billing_plan, monthly_limit, queries, energy, avg_time = row
                    usage_percentage = (queries / monthly_limit * 100) if monthly_limit > 0 else 0
                    
                    tenants_summary.append({
                        "tenant_id": tenant_id,
                        "business_name": business_name,
                        "billing_plan": billing_plan,
                        "monthly_limit": monthly_limit,
                        "total_queries": queries,
                        "total_energy": round(energy, 4),
                        "avg_response_time": round(avg_time, 2) if avg_time else 0,
                        "usage_percentage": round(usage_percentage, 2)
                    })
                    
                    total_queries += queries
                    total_energy += energy
                
                return {
                    "tenants": tenants_summary,
                    "summary": {
                        "total_tenants": len(tenants_summary),
                        "total_queries": total_queries,
                        "total_energy": round(total_energy, 4),
                        "avg_queries_per_tenant": round(total_queries / len(tenants_summary), 2) if tenants_summary else 0
                    }
                }
                
        except Exception as e:
            logger.error(f"Get tenants summary error: {e}")
            return None

# Global SQLite service instance - lazy loading
sqlite_service = None

def get_sqlite_service():
    """Lazy loading için SQLite service getter"""
    global sqlite_service
    if sqlite_service is None:
        sqlite_service = SQLiteDatabaseService()
    return sqlite_service

# --- Pydantic Modelleri ---
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    tenant_id: int

class NLUSingleAnalysis(BaseModel):
    slm_intent: str
    slm_entities: List[Dict[str, Any]]
    confidence_score: float
    message: str

class NLUResponse(BaseModel):
    original_query: str
    processed_query_for_nlu: Optional[str] = None
    session_id: str
    tenant_id: int
    nlu_method: str
    analysis: Optional[NLUSingleAnalysis] = None
    detected_intent: Optional[str] = None
    previous_query_in_session: Optional[str] = None
    resolved_item_details: Optional[Dict[str, Any]] = None
    resolved_size: Optional[str] = None
    actionable_message: Optional[str] = None
    bot_response: Optional[str] = None
    ask_for_clarification: bool = False
    clarification_options: Optional[List[Dict[str, Any]]] = None

# --- Global Kaynaklar ve Ayarlar ---
MODEL_PATH = BASE_DIR / "nlu_model.bin"
MODEL_PATH_FALLBACK = (BASE_DIR / "nlu_model.bin").resolve()
TURKISH_FREQUENCY_DICTIONARY_PATH = BASE_DIR / "turkish_frequency_dictionary.txt"

# NLU Ayarları ve Kuralları
product_extraction_intents = ["fiyat_sorgulama", "ürün_bilgisi_sorma", "stok_sorgulama", "iade_sorgulama", "ürün_malzeme_sorma"]

# Zamir tespiti için regex
pronoun_patterns = {
    "bu": re.compile(r"\b(bu|şu|o)\s+(ürün|pantolon|gömlek|ceket|elbise|etek|ayakkabı|model|item)\b", re.IGNORECASE),
    "bu_genel": re.compile(r"\b(bu|şu|o)\s+(fiyat|stok|beden|malzeme|içerik|özellik|renk|numara)\b", re.IGNORECASE),
    "bunun": re.compile(r"\b(bunun|şunun|onun)\s+(fiyatı|stoku|bedeni|malzemesi|içeriği|özelliği|rengi|numarası)\b", re.IGNORECASE),
    "bundan": re.compile(r"\b(bundan|şundan|ondan)\s+(istiyorum|alabilir miyim|beğendim)\b", re.IGNORECASE),
    "bunda": re.compile(r"\b(bunda|şunda|onda)\s+(renk|seçenek|farklı|başka)\b", re.IGNORECASE),
    "bunu": re.compile(r"\b(bunu|şunu|onu)\s+(beğendim|alabilir miyim|istiyorum)\b", re.IGNORECASE),
    "sunu": re.compile(r"\b(sunu|bunu|onu)\s+(alabilir miyim|istiyorum|beğendim)\b", re.IGNORECASE),
    "o_ne": re.compile(r"\bo\s+ne\s+(kadar|fiyatı|ederi)\b", re.IGNORECASE),
    "peki_ya": re.compile(r"\b(peki ya|ya da|veya)\s+(kırmızısı|mavisi|yeşili|siyahı)\b", re.IGNORECASE),
}

# GÜÇLENDIRILMIŞ REGEX RULES - "var mı?" sorunlarını çözmek için
rules = {
    "ürün_arama": re.compile(r"\b(gecelik|pijama|pjama|sabahlık|sabalik|abiye|tesettür|günlük\s+elbise|kışlık\s+mont|elbise|tulum|şort|kapri|takım|takımı|çorap|boxer|sütyen|iç\s+çamaşırı|dantelli|saten|kadife|brode|dekolteli|düğmeli|askılı)\s*.*\b(var\s+m[ıi]|varm[ıi]|var\s+mi|vr\s+m[ıi]|vr|varmi|varmı|mevcut\s+mu|bulunur\s+mu|satıyor\s+musunuz|arıyorum|lazım|göster|ara|bul|listele)\b", re.IGNORECASE),
    "stok_sorgulama": re.compile(r"\b(\d{2,3}|s|m|l|xl|xs|xxl|xxxl|small|medium|large|xlarge|numara|beden|bedeni|bedenleri|stok|stokta|stokda)\s*.*\b(var\s+m[ıi]|varm[ıi]|var\s+mi|vr\s+m[ıi]|vr|varmi|varmı|mevcut\s+mu|bulunur\s+mu|kaldı\s+m[ıi]|stokta\s+m[ıi]|stokda\s+m[ıi])\b", re.IGNORECASE),
    "renk_stok_sorgulama": re.compile(r"\b(siyah|beyaz|kırmızı|kirmizi|mavi|yeşil|yesil|sarı|sari|pembe|mor|turuncu|kahverengi|gri|lacivert|bordo)\s*(.*?)\s*\b(var\s+m[ıi]|varm[ıi]|var\s+mi|vr\s+m[ıi]|vr|varmi|varmı|mevcut\s+mu|bulunur\s+mu)\b", re.IGNORECASE),
    "calisma_saatleri_sorma": re.compile(r"\b((?:çalışma|calisma)\s+saatleri(?:niz)?(?:[\s,.]*nedir\??)?|kaça\s+kadar\s+açık|ne\s+zaman\s+açık|açılış\s+kapanış|mesai|hafta\s*sonu\s+açık|pazar\s+açık\s*mı|hangi\s+saatler|ne\s+zaman\s+kapanıyor|saat\s+kaçta\s+açılıyor|saat\s+kaçta\s+kapanır|açıksınız|calisma\s+saati)\b", re.IGNORECASE),
    "kargo_bilgisi_sorma": re.compile(r"\b(kargo|gönderim|teslimat|kaç günde gelir|kargo ücret|kargo ne kadar|kargo takip|yurtiçi kargo|kargo nekadar|kargonuz kaç günde|teslim süresi|kargo tutar|sipariş nasıl|ne kadar sürer|takip edebilir miyim|kargo firması|yurt dışı|ücretsiz kargo|gelmedi|gecikme|kargo ücreti|kargo ücreti ne kadar)\b", re.IGNORECASE),
    "fiyat_sorgulama": re.compile(r"\b(fiyat|ücret|kaç\s+para|ne\s+kadar|kaç\s+tl|maliyet|ederi|kaça|nekadar|fyt|fiyt|fyaat|fiyay|kça\s+para|ne\s+kadr|fiyatı\s+ne|fiyatı\s+nedir|ücreti\s+nedir|fiyatını\s+öğren|fiyat\s+bilgisi)(?!.*(?:kargo|teslimat|gönderim|açık|kapanış|saatler\w*|saat|iade|stok|malzeme|özellik|beden|nerede|adres|konum|telefon|mail|ödeme|site|çalışma|calisma|kumaş|içerik)\b)\b", re.IGNORECASE),
    "selamlama": re.compile(r"^\s*(merhaba|selam|iyi\s+günler|günaydın|mrb|slm|sa|selamun\s+aleykum|hey|kolay\s+gelsin|merhba|gunaydn|selamlarr|meraba|s\.a\.|nbr|heyo|selamlar|iyi\s+akşamlar|hayırlı\s+işler)\b", re.IGNORECASE),
    "iade_sorgulama": re.compile(r"\b(iade|geri verme|değişim|değiştir|iade edebilir|iade koşul.*|koşullaeı|para iadesi|değiştirebilir miyim|geri gönderebilir|ürünü geri al|beğenmedim|nasıl yapılır|kaç günde|14 gün|defolu|kusurlu|yanlış ürün|eksik|iptal|iade var mı|iade yapıyor musunuz)\b", re.IGNORECASE),
    "tesekkur": re.compile(r"^\s*(teşekkür|sağ\s*ol|tşk|eyvallah|saol|mersi|eyw|tskler|tamam|tmm|ok|anladım|pekala|tamamdır|varol)\b", re.IGNORECASE),
    "ürün_malzeme_sorma": re.compile(r"\b(malzeme|içeriğ\w*|kumaş\w*|astar|yapılmış|üretilmiş|neyden\s+yapıl|materyal|kumas\s+ne|kompozisyonu)(?!.*(?:stok|beden|fiyat|kaç\s+para|ne\s+kadar|bilgi|özellik)\b)\b", re.IGNORECASE),
    "ürün_bilgisi_sorma": re.compile(r"\b(özellik\w*|hakkında\s+bilgi|detay|açıklama|nedir\s+bu|ne\s+işe\s+yarar|ürün\s+bilgisi|ürünle\s+ilgili|model\s+hakkında|ürün\s+ne\s+için|anlatır\s+mısın\s+bu\s+ürün|spesifikasyonları)(?!.*(?:malzeme|kumaş|içerik)\b)\b", re.IGNORECASE),
    "lokasyon_sorma": re.compile(r"\b(mağaza|adres|nerede|konum|yer|lokasyon|adresiniz|mağazanız|hangi şehir|hangi il|hangi semt|hangi mahalle|hangi cadde|hangi sokak)\b", re.IGNORECASE),
    "tel_no_sorma": re.compile(r"\b(telefon|tel\s+no|numara|iletişim\s+no|arayabilir|whatsapp|mail|e-posta|eposta|numaranız|mail\s+adresiniz|irtibat)\b", re.IGNORECASE),
    "odeme_yontemleri_sorma": re.compile(r"\b(nasıl\s+öde|ödeme\s+seçenek|ne\s+kabul|kredi\s+kartı|taksit|kapıda\s+ödeme|havale|eft|ödeme\s+türleri|ödeme\s+yapabilir|taksit\s+imkanı|ödeme\s+şekilleri)(?!.*(?:stok|beden)\b)\b", re.IGNORECASE),
    "websitesi_sorma": re.compile(r"\b(web|site|instagram|facebook|link|url|adres|sosyal medya|online|internet|web siteniz|instagram hesabınız|facebook sayfanız|linkiniz|sosyal medya hesabınız)\b", re.IGNORECASE),
    "musteri_hizmetlerine_baglanma": re.compile(r"\b(müşteri\s+hizmet|yetkili\s+biri|canlı\s+destek|insanla\s+konuş|temsilciye\s+aktar|operatöre\s+bağlan|birine\s+bağla)\b", re.IGNORECASE),
    "siparis_durumu_sorma": re.compile(r"\b(siparişim\s+ne\s+durumda|kargom\s+nerede|siparişimi\s+takip|kargom\s+ne\s+zaman\s+gelir|sipariş\s+no\s+.*\s+ne\s+oldu|ürünüm\s+gelmedi|kargo\s+gelmedi|sipariş\s+durumu)\b", re.IGNORECASE),
    "oneri_isteme": re.compile(r"\b(ne\s+önerirsin|tavsiye\s+eder|en\s+çok\s+satan|benzer\s+ne\s+var|alternatif\s+ne|öneri\s+var\s+mı|ne\s+tavsiye|bir\s+şey\s+öner|hangi\s+ürünü\s+almalı|ne\s+seçmeli)\b", re.IGNORECASE),
    "olumsuz_yanıt": re.compile(r"^\s*(hayır|yok\s+kalsın|gerek\s+yok|istemiyorum|düşünmüyorum|pas|vazgeçtim|kalsın|olmaz|hayr|ilgilenmiyorum|almayayım)\b", re.IGNORECASE),
}
GENERAL_INTENTS_FOR_OVERRIDE = ["selamlama", "tesekkur", "olumsuz_yanıt"]
MIN_WORDS_FOR_SLM_OVERRIDE = 2
SLM_OVERRIDE_CONFIDENCE_THRESHOLD = 0.60
FUZZY_MATCH_THRESHOLD = 60

# --- Kelime Listeleri ---
PROTECTED_WORDS_SYMSPELL = { "mrb", "slm", "tşk", "eyw", "tmm", "ok", "sa", "kot", "fiyat", "stok", "beden", "ürün", "urun", "s", "m", "l", "xl", "xs", "xxl", "xxxl", "small", "medium", "large", "xlarge", "bu", "şu", "o", "ne", "mi", "var", "yok", "kaç", "gibi", "göre", "kadar", "için", "ile", "ve", "veya", "ya da", "tl", "try", "pantolon", "gömlek", "ceket", "elbise", "etek", "ayakkabı", "model", "kırmızı", "mavi", "yeşil", "sarı", "siyah", "beyaz", "pembe", "mor", "turuncu", "gri", "kahverengi", "rengi", "fiyatlar", "fiyatları", "bedenler", "bedenleri", "bedeninde", "stokta", "stokları", "stoklar", "nedir", "acaba", "miyim", "musunuz", "varmi", "varmı", "mevcutmu", "mevcut", "kaldı", "öğrenebilir", "değiştirebilir", "edebilir", "söyler", "alabilir", "olabilir", "yapabilir", "ipek", "deri", "keten", "para", "imkanı", "koşulları", "adresiniz", "saatleriniz", "konumunuz", "sizde", "sizden", "bana", "sana", "ona", "ol", "hayat"}
SYMSPELL_BLOCKED_CORRECTIONS = { ("ceketin", "çektin"), ("kot", "koy"), ("Kot", "not"), ("fiyatını", "hayatını"), ("fiyatını", "hayat"), ("medium", "demek"), ("stokta", "nokta"), ("stok", "sokmak"), ("urun", "uzun"), ("ürünün", "uzunun"), ("modelin", "modemin"), ("bednleri", "bebekleri"), ("ceketler", "cesetler"), ("tl", "ol"), ("kumas", "kuma"), ("imkanı", "mekanı"), ("ipek", "i pek"), ("ipek", "i̇pek"), ("ipek", "ek"), ("ipek", "i"), ("para", "par")}
KNOWN_TYPOS = { "pantoon": "pantolon", "pntolon": "pantolon", "pantalon": "pantolon", "pantln": "pantolon", "jekt": "ceket", "jeket": "ceket", "ceketler": "ceket", "kadr": "kadar", "nedr": "nedir", "bedn": "beden", "bednleri": "bedenleri", "kırmzı": "kırmızı", "fiyay": "fiyat", "fyt": "fiyat", "kça": "kaça", "fiyt": "fiyat", "calısma": "çalışma", "magazanız": "mağazanız", "ürnün": "ürünün", "urun": "ürün", "koşullaeı": "koşulları", "merhba": "merhaba", "gunaydn": "günaydın", "selamlarr": "selamlar", "meraba": "merhaba", "s.a.": "selamün aleyküm", "nbr": "ne haber", "smal": "small", "ktene": "keten", "ketn": "keten", "stokda": "stokta", "varmi": "var mı", "mevcutmu": "mevcut mu", "i̇pek": "ipek", "i pek": "ipek", "i ek": "ipek", "iek": "ipek", "fiyati": "fiyatı", "taksi": "taksit", "ol": "tl"}
CORE_WORDS_TO_PRESERVE_LEMMA = { "fiyat", "stok", "beden", "iade", "kargo", "adres", "konum", "telefon", "mail", "ödeme", "site", "kumaş", "malzeme", "içerik", "özellik", "sipariş", "taksit", "indirim", "kampanya", "ücret", "para", "pantolon", "gömlek", "ceket", "elbise", "etek", "ayakkabı", "model", "ürün", "urun", "s", "m", "l", "xl", "xs", "xxl", "xxxl", "small", "medium", "large", "kırmızı", "mavi", "yeşil", "sarı", "siyah", "beyaz", "pembe", "mor", "turuncu", "gri", "kahverengi", "renk", "rengi", "kaç", "ne", "nasıl", "nedir", "kadar", "var", "yok", "bu", "şu", "o", "acaba", "miyim", "misin", "mı", "mi", "mu", "mü", "öğrenebilir", "değiştirebilir", "edebilir", "söyler", "mevcut", "kaldı", "alabilir", "olabilir", "yapabilir", "fiyatlar", "bedenler", "stoklar", "bedeninde", "stokta", "numara", "numarası", "no", "tl", "try", "imkan", "imkanı", "koşul", "koşulları", "ipek", "deri", "keten", "kot"}

# --- Uygulama Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('LIFESPAN: Başladı')
    print('LIFESPAN: Başladı')
    try:
        app.state.sqlite_service = get_sqlite_service()
        logger.info('LIFESPAN: SQLite OK')
        print('LIFESPAN: SQLite OK')
    except Exception as e:
        logger.error(f'LIFESPAN: SQLite HATA: {e}')
        print(f'LIFESPAN: SQLite HATA: {e}')
    # FastText - AKTİF
    try:
        global current_nlu_model
        model_path = BASE_DIR / "nlu_model.bin"
        if model_path.exists():
            current_nlu_model = fasttext.load_model(str(model_path))
            logger.info('LIFESPAN: FastText OK')
            print('LIFESPAN: FastText OK')
        else:
            logger.warning('LIFESPAN: FastText modeli bulunamadı')
            print('LIFESPAN: FastText modeli bulunamadı')
    except Exception as e:
        logger.error(f'LIFESPAN: FastText HATA: {e}')
        print(f'LIFESPAN: FastText HATA: {e}')
    # Zeyrek - AKTİF
    try:
        global analyzer
        analyzer = MorphAnalyzer()
        logger.info('LIFESPAN: Zeyrek OK')
        print('LIFESPAN: Zeyrek OK')
    except Exception as e:
        logger.error(f'LIFESPAN: Zeyrek HATA: {e}')
        print(f'LIFESPAN: Zeyrek HATA: {e}')
    # SymSpell - AKTİF
    try:
        global spell_checker
        spell_checker = SymSpell()
        spell_checker.load_dictionary(str(BASE_DIR / "turkish_frequency_dictionary.txt"), term_index=0, count_index=1)
        logger.info('LIFESPAN: SymSpell OK')
        print('LIFESPAN: SymSpell OK')
    except Exception as e:
        logger.error(f'LIFESPAN: SymSpell HATA: {e}')
        print(f'LIFESPAN: SymSpell HATA: {e}')
    app.state.critical_resources_loaded = True
    app.state.lifespan_was_executed = True
    app.state.fast_mode_active = FAST_MODE
    logger.info('LIFESPAN: yield öncesi')
    print('LIFESPAN: yield öncesi')
    yield
    logger.info('LIFESPAN: yield sonrası')
    print('LIFESPAN: yield sonrası')

print('--- FastAPI app tanımı öncesi ---')
app = FastAPI(lifespan=lifespan)
print('--- FastAPI app tanımı sonrası ---')

@app.get("/ping")
async def ping():
    print('PING endpoint çağrıldı')
    return {"status": "ok"}

# --- CORS Middleware ---
origins = ["http://localhost", "http://localhost:8000", "null"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NLU Yardımcı Fonksiyonları ---
def _normalize_for_match(text: str) -> str:
    if not text: return ""
    return text.strip().lower().replace("i̇", "i")

def _detect_pronoun_usage(query: str) -> Dict[str, Any]:
    """Zamir kullanımını tespit eder ve bağlam gereksinimini belirler."""
    query_lower = query.lower()
    
    # Genişletilmiş zamir desenleri
    pronoun_patterns = {
        "bu": re.compile(r"\b(bu|şu|o)\s+(ürün|pantolon|gömlek|ceket|elbise|etek|ayakkabı|model|item)\b", re.IGNORECASE),
        "bu_genel": re.compile(r"\b(bu|şu|o)\s+(fiyat|stok|beden|malzeme|içerik|özellik|renk|numara)\b", re.IGNORECASE),
        "bunun": re.compile(r"\b(bunun|şunun|onun)\s+(fiyatı|stoku|bedeni|malzemesi|içeriği|özelliği|rengi|numarası)\b", re.IGNORECASE),
        "bundan": re.compile(r"\b(bundan|şundan|ondan)\s+(istiyorum|alabilir miyim|beğendim)\b", re.IGNORECASE),
        "bunda": re.compile(r"\b(bunda|şunda|onda)\s+(renk|seçenek|farklı|başka)\b", re.IGNORECASE),
        "bunu": re.compile(r"\b(bunu|şunu|onu)\s+(beğendim|alabilir miyim|istiyorum)\b", re.IGNORECASE),
        "sunu": re.compile(r"\b(sunu|bunu|onu)\s+(alabilir miyim|istiyorum|beğendim)\b", re.IGNORECASE),
        "o_ne": re.compile(r"\bo\s+ne\s+(kadar|fiyatı|ederi)\b", re.IGNORECASE),
        "peki_ya": re.compile(r"\b(peki ya|ya da|veya)\s+(kırmızısı|mavisi|yeşili|siyahı)\b", re.IGNORECASE),
    }
    
    # Tüm zamir desenlerini kontrol et
    for pattern_name, pattern in pronoun_patterns.items():
        match = pattern.search(query_lower)
        if match:
            return {
                "has_pronoun": True,
                "pronoun_type": pattern_name,
                "requires_context": True,
                "matched_text": match.group()
            }
    
    # Basit zamir kontrolü (daha geniş)
    simple_pronouns = ["bu", "şu", "o", "bunu", "şunu", "onu", "bundan", "şundan", "ondan", "bunda", "şunda", "onda"]
    words = query_lower.split()
    for word in words:
        if word in simple_pronouns:
            return {
                "has_pronoun": True,
                "pronoun_type": "simple",
                "requires_context": True,
                "matched_text": word
            }
    
    return {
        "has_pronoun": False,
        "pronoun_type": None,
        "requires_context": False,
        "matched_text": None
    }

def _safe_lemmatize_word(word: str, current_morphology: Optional[zeyrek.MorphAnalyzer]) -> str:
    if not current_morphology: return word.lower()
    word_lower = word.lower().strip()
    if not word_lower or word_lower.isdigit(): return word_lower
    if word_lower in {"i̇pek", "ipek", "iek"}: return "ipek"
    if word_lower in {"tl", "para", "urun", "stok", "medium", "taksit"}: return word_lower
    if word_lower in CORE_WORDS_TO_PRESERVE_LEMMA: return word_lower
    plural_map = {"ceketler": "ceket", "gömlekler": "gömlek", "pantolonlar": "pantolon", "bedenler": "beden", "fiyatlar": "fiyat", "koşullar": "koşul", "özellikler": "özellik", "malzemeler": "malzeme", "renkler": "renk", "stoklar": "stok"}
    if word_lower in plural_map: return plural_map[word_lower]
    product_bases_for_possessive = ["fiyat", "beden", "stok", "model", "ürün", "gömlek", "ceket", "pantolon", "kumaş", "malzeme", "içerik", "özellik", "renk", "adres", "konum", "telefon", "imkan", "koşul", "numara"]
    for base in product_bases_for_possessive:
        if word_lower.startswith(base):
            if len(word_lower) == len(base) + 1 and word_lower[-1] in "ıiuü": return base
            if len(word_lower) == len(base) + 2 and word_lower[-2:] in ["sı", "si", "su", "sü", "ın", "in", "un", "ün"]: return base
            if len(word_lower) == len(base) + 3 and word_lower[-3:] in ["ını", "ini", "unu", "ünü"]: return base
    analysis_results = current_morphology.analyze(word_lower)
    if analysis_results and analysis_results[0] and analysis_results[0][0]:
        best_analysis = analysis_results[0][0]
        lemma = best_analysis.lemma.lower()
        pos = best_analysis.pos
        if lemma == "unk" or "Unk" in pos or "Punc" in pos: return word_lower
        if pos not in ["Verb", "Adverb"]:
            if pos in ["Prop", "Abbrv", "Num", "Conj", "Interj", "Postp", "Det", "Ques"]: return word_lower
            if len(lemma) < 3 and word_lower != lemma and not word_lower.isdigit(): return word_lower
            if pos in ["Noun", "Adjective"] and len(lemma) < len(word_lower) - 3 and fuzz.ratio(word_lower, lemma) < 60:
                return word_lower
        return lemma
    return word_lower

def _preprocess_text_for_matching(text_phrase: str, current_morphology: Optional[zeyrek.MorphAnalyzer]) -> str:
    if not text_phrase or not text_phrase.strip(): return ""
    lower_text = text_phrase.lower().strip().replace("i̇", "i")
    lower_text = re.sub(r"\bi\s+(pek|ek)\b", "ipek", lower_text)
    cleaned_text = re.sub(r"[,\.!?\";:]", " ", lower_text)
    cleaned_text = re.sub(r"['`'']", "", cleaned_text)
    cleaned_text = re.sub(r"[^\w\s]", " ", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    words = cleaned_text.split()
    if not current_morphology: return " ".join(words)
    lemmatized_words = [_safe_lemmatize_word(word, current_morphology) for word in words if word]
    return " ".join(lemmatized_words).strip()

def correct_spelling(text: str, current_sym_spell: Optional[SymSpell]) -> str:
    if not current_sym_spell or not text: return text
    words = text.split()
    corrected_words = []
    for word in words:
        word_lower = word.lower()
        if word_lower in PROTECTED_WORDS_SYMSPELL or word_lower in KNOWN_TYPOS:
            corrected_words.append(word)
            continue
        if word_lower in KNOWN_TYPOS:
            corrected_words.append(KNOWN_TYPOS[word_lower])
            continue
        suggestions = current_sym_spell.lookup(word_lower, Verbosity.CLOSEST, max_edit_distance=1)
        if suggestions and suggestions[0].term != word_lower:
            corrected_term = suggestions[0].term
            if (word_lower, corrected_term) not in SYMSPELL_BLOCKED_CORRECTIONS:
                corrected_words.append(corrected_term)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

def preprocess_query_for_nlu(query: str, current_sym_spell: Optional[SymSpell], current_morphology: Optional[zeyrek.MorphAnalyzer]) -> str:
    spell_checked = correct_spelling(query, current_sym_spell)
    return _preprocess_text_for_matching(spell_checked, current_morphology)

def call_slm_model(processed_query: str, current_nlu_model: Optional[fasttext.FastText._FastText]) -> NLUSingleAnalysis:
    if not current_nlu_model or not processed_query:
        return NLUSingleAnalysis(slm_intent="kapsam_disi", slm_entities=[], confidence_score=0.0, message="Model yüklenemedi")
    
    try:
        prediction = current_nlu_model.predict(processed_query, k=1, threshold=0.1)
        if prediction[0] and len(prediction[0]) > 0:
            intent_label = prediction[0][0].replace("__label__", "")
            confidence = prediction[1][0] if prediction[1] and len(prediction[1]) > 0 else 0.0
            return NLUSingleAnalysis(slm_intent=intent_label, slm_entities=[], confidence_score=confidence, message=f"SLM tahmin: {intent_label} ({confidence:.3f})")
    except Exception as e:
        logger.error(f"FastText tahmin hatası: {e}")
    
    return NLUSingleAnalysis(slm_intent="kapsam_disi", slm_entities=[], confidence_score=0.0, message="SLM tahmin yapamadı")

def extract_simple_entities(original_query_spell_checked: str, processed_query_lemmatized: str,
                            current_morphology: Optional[zeyrek.MorphAnalyzer],
                            intent: Optional[str] = None) -> Dict[str, Any]:
    entities = {"item_name_candidate": "", "size": None}  # Boş string olarak başlat
    
    if not original_query_spell_checked: return entities
    
    query_lower = original_query_spell_checked.lower()
    
    # Beden tespiti - daha kapsamlı
    size_patterns = [
        (r"\b(s|m|l|xl|xs|xxl|xxxl)\b", lambda m: m.group(1).upper()),  # Büyük harfe çevir
        (r"\b(small|medium|large|xlarge)\b", lambda m: m.group(1).upper()),  # Büyük harfe çevir
        (r"\b(\d{2,3})\s*(cm|inch|inç|numara)\b", lambda m: m.group(1)),  # Numara tespiti
        (r"\b(\d{2,3})\b", lambda m: m.group(1))  # Sadece numara
    ]
    
    for pattern, formatter in size_patterns:
        size_match = re.search(pattern, query_lower)
        if size_match:
            entities["size"] = formatter(size_match)
            break
    
    # Ürün adı tespiti - daha gelişmiş yaklaşım
    product_keywords = ["pantolon", "gömlek", "ceket", "elbise", "etek", "ayakkabı"]
    material_keywords = ["ipek", "deri", "keten", "kot"]
    
    words = query_lower.split()
    
    # Önce malzeme+ürün kombinasyonlarını ara
    found_products = []
    
    # İki kelimelik kombinasyonları kontrol et
    for i in range(len(words) - 1):
        if words[i] in material_keywords and words[i+1] in product_keywords:
            product_phrase = f"{words[i]} {words[i+1]}"
            found_products.append(product_phrase)
    
    # Eğer kombinasyon bulunamadıysa, çekim eklerini de kontrol et
    if not found_products:
        for i in range(len(words) - 1):
            if words[i] in material_keywords:
                # Sonraki kelime ürün kelimesi ile başlıyor mu kontrol et
                for product in product_keywords:
                    if words[i+1].startswith(product):
                        product_phrase = f"{words[i]} {product}"
                        found_products.append(product_phrase)
                        break
    
    # Eğer hala bulunamadıysa, tek kelimeleri ara
    if not found_products:
        for word in words:
            if word in product_keywords:
                found_products.append(word)
            elif word in material_keywords:
                found_products.append(word)
            else:
                # Çekim eklerini kontrol et
                for product in product_keywords:
                    if word.startswith(product):
                        found_products.append(product)
                        break
    
    if found_products:
        # En uzun ürün adını seç
        entities["item_name_candidate"] = max(found_products, key=len)
    
    return entities

def _detect_multiple_questions(query: str) -> List[str]:
    """Sorguda birden fazla soru olup olmadığını tespit eder ve ayrıştırır."""
    query_lower = query.lower()
    
    # Çoklu soru belirteçleri
    multiple_question_indicators = [
        r"\bve\b",
        r"\bbir de\b", 
        r"\bbide\b",
        r"\bhem\b",
        r"\bhem de\b",
        r"\bayrıca\b",
        r"\bda\b",
        r"\bde\b"
    ]
    
    # Çoklu soru var mı kontrol et
    has_multiple = any(re.search(pattern, query_lower) for pattern in multiple_question_indicators)
    
    if not has_multiple:
        return [query]
    
    # Soruları ayrıştır
    questions = []
    
    # "ve" ile ayrılmış sorular
    if " ve " in query_lower:
        parts = query.split(" ve ")
        if len(parts) > 1:
            questions.extend([part.strip() for part in parts if part.strip()])
    
    # "bir de" ile ayrılmış sorular
    elif "bir de" in query_lower or "bide" in query_lower:
        parts = re.split(r"\b(bir de|bide)\b", query, flags=re.IGNORECASE)
        if len(parts) > 2:
            questions.append(parts[0].strip())
            questions.append(parts[2].strip())
    
    # "hem" ile ayrılmış sorular
    elif "hem" in query_lower:
        parts = re.split(r"\bhem\b", query, flags=re.IGNORECASE)
        if len(parts) > 1:
            questions.extend([part.strip() for part in parts if part.strip()])
    
    # Eğer ayrıştırma başarısız olursa orijinal sorguyu döndür
    if not questions:
        return [query]
    
    return questions

def _generate_multiple_response(questions: List[str], responses: List[str]) -> str:
    """Çoklu sorulara numaralı cevap oluşturur."""
    if len(questions) != len(responses):
        return responses[0] if responses else "Merhaba! Sorularınızı yanıtlamaya çalışıyorum."
    
    response_parts = []
    for i, (question, response) in enumerate(zip(questions, responses), 1):
        response_parts.append(f"{i}. {response}")
    
    return "\n".join(response_parts)

# --- Ana Sorgu İşleme Endpoint'i ---
@app.post("/process_query/", response_model=NLUResponse)
async def process_query(payload: QueryRequest, request: FastAPIRequest):
    start_time = datetime.now(timezone.utc)
    request_start_time = time.time()
    original_user_query = payload.query
    tenant_id = payload.tenant_id
    effective_session_id = payload.session_id or str(uuid.uuid4())

    try:
        # MVP: Rate limiting kontrolü
        rate_limit_ok, rate_limit_message = check_rate_limit(tenant_id)
        if not rate_limit_ok:
            logger.warning(f"Rate limit exceeded for tenant {tenant_id}: {rate_limit_message}")
            return NLUResponse(
                original_query=original_user_query,
                session_id=effective_session_id,
                tenant_id=tenant_id,
                nlu_method="rate_limited",
                detected_intent="rate_limited",
                actionable_message=f"Günlük query limitiniz doldu. Lütfen yarın tekrar deneyin. ({rate_limit_message})",
                bot_response=f"Günlük query limitiniz doldu. Lütfen yarın tekrar deneyin.",
                ask_for_clarification=False
            )

        # Modeldan tahmin al ve conflict resolution yap
        
        # Cache kontrolü önce
        cached_result = get_cached_intent(original_user_query)
        
        if cached_result:
            detected_intent, confidence, nlu_method = cached_result
            logger.info(f"💾 Cache hit: {detected_intent} ({confidence:.3f})")
            analysis = None
            energy_consumption = 0.05  # Cache çok düşük enerji
        elif True:  # ZORLA FAST_MODE - ML tamamen bypass
            # FAST MODE FORCED - Sadece regex kullan, ML bypass
            logger.info("⚡ FAST MODE FORCED: ML TAMAMEN BYPASS - Sadece regex")
            detected_intent, confidence = detect_intent_fast_mode(original_user_query)
            nlu_method = "fast_mode_regex_only"
            analysis = None
            energy_consumption = 0.1  # Regex çok düşük enerji
            # Cache'e kaydet
            cache_intent_result(original_user_query, detected_intent, confidence, nlu_method)
        else:
            # ML + regex hibrit yaklaşım - ÜRÜN ARAMA ÖNCELİKLİ VERSİYON
            logger.info("🤖 Gelişmiş Hibrit Mode: ML + Regex + Context")
            
            # 0. ÖNCELİK: Ürün arama pattern kontrolü (ML'den önce!)
            query_lower = original_user_query.lower()
            product_search_patterns = [
                r"\b(gecelik|pijama|abiye|tesettür|günlük\s+elbise|kışlık\s+mont)\s+.*(var\s+m[ıi]|varm[ıi]|mevcut\s+mu|bulunur\s+mu)\b",
                r"\b(gecelik|pijama|abiye|tesettür)\s+(var|vr|varmi|varmı|mevcut|bulunur)\b"
            ]
            
            is_product_search = any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in product_search_patterns)
            
            if is_product_search:
                # DOĞRUDAN ürün arama - ML'yi bypass et
                detected_intent = "ürün_arama"
                confidence = 0.95  # Çok yüksek güven
                nlu_method = "hybrid_pattern_override"
                analysis = None
                energy_consumption = 0.1  # Çok düşük enerji (pattern matching)
                logger.info(f"🎯 Pattern Override: {original_user_query} -> ürün_arama")
            else:
                # 1. ML ile intent tespiti
                processed_query = preprocess_query_for_nlu(original_user_query, current_sym_spell, current_morphology)
                ml_analysis = call_slm_model(processed_query, current_nlu_model)
                
                # 2. Regex ile intent tespiti
                regex_intent, regex_confidence = detect_intent_fast_mode(original_user_query)
                
                # 3. Hibrit karar verme
                if regex_confidence > 0.7 and regex_intent == "ürün_arama":
                    # Regex ürün arama tespit etti, güvenilir
                    detected_intent = regex_intent
                    confidence = regex_confidence
                    nlu_method = "hybrid_regex_primary"
                    energy_consumption = 0.2  # Regex düşük enerji
                elif ml_analysis.confidence_score > 0.8 and ml_analysis.slm_intent not in ["indirim_kampanya_sorma"]:
                    # ML çok güvenilir ama problematik intent'ler değil
                    detected_intent = ml_analysis.slm_intent
                    confidence = ml_analysis.confidence_score
                    nlu_method = "hybrid_ml_primary"
                    energy_consumption = 0.8  # ML yüksek enerji
                elif regex_confidence > 0.5:
                    # Regex güvenilir, regex sonucunu kullan
                    detected_intent = regex_intent
                    confidence = regex_confidence
                    nlu_method = "hybrid_regex_preferred"
                    energy_consumption = 0.2  # Regex düşük enerji
                else:
                    # Context-aware fallback
                    detected_intent = resolve_intent_conflict(original_user_query, ml_analysis, regex_intent, regex_confidence)
                    confidence = max(ml_analysis.confidence_score, regex_confidence) if ml_analysis else regex_confidence
                    nlu_method = "hybrid_context_fallback"
                    energy_consumption = 0.4  # Context analysis orta enerji
                
                analysis = ml_analysis
            
            # Cache'e kaydet
            cache_intent_result(original_user_query, detected_intent, confidence, nlu_method)
        
        # Entity extraction
        if current_morphology:
            # Gelişmiş entity extraction
            entities = extract_simple_entities(original_user_query, "", current_morphology, detected_intent)
            energy_consumption += 0.1  # Entity extraction enerji
        else:
            # Basit entity extraction
            entities = {"item_name_candidate": "", "size": None}
        
        # Tenant settings al
        tenant_settings = await get_sqlite_service().get_tenant_settings(tenant_id) or {}
        
        # Session context'ini getir
        session_context = get_session_context(effective_session_id, tenant_id)
        
        # Response generation
        if FAST_MODE:
            response, clarification_needed = generate_response_fast_mode(detected_intent, entities, original_user_query, tenant_settings)
        else:
            # Gelişmiş response generation
            response, clarification_needed = generate_response_hybrid(detected_intent, entities, original_user_query, tenant_settings, analysis)
            energy_consumption += 0.1  # Response generation enerji

        # Session management
        session_data = await get_sqlite_service().get_session_data(effective_session_id, tenant_id) or {}
        current_turn_log = { "query": original_user_query, "bot_response": response, "timestamp": datetime.now(timezone.utc).isoformat() }
        session_history = session_data.get("history", [])
        session_history.append(current_turn_log)
        session_data["history"] = session_history[-5:]
        await get_sqlite_service().save_session_data(effective_session_id, tenant_id, session_data)
        
        # Response time hesapla
        response_time = time.time() - request_start_time
        
        # MVP: Tenant usage tracking
        await get_sqlite_service().update_tenant_usage(
            tenant_id=tenant_id,
            response_time=response_time,
            energy_consumption=energy_consumption,
            nlu_method=nlu_method,
            success=True
        )
        
        # Analytics logging
        log_analytics(detected_intent, response_time, success=True)
        
        # Session context'ini güncelle
        update_session_context(effective_session_id, tenant_id, original_user_query, detected_intent, response)
        
        return NLUResponse(
            original_query=original_user_query,
            processed_query_for_nlu=original_user_query,  # FAST_MODE'da da aynı query kullan
            session_id=effective_session_id,
            tenant_id=tenant_id,
            nlu_method=nlu_method,
            analysis=analysis,
            detected_intent=detected_intent,
            previous_query_in_session=session_history[-2]["query"] if len(session_history) > 1 else None,
            resolved_item_details=None,
            resolved_size=entities.get("size"),
            actionable_message=response,
            bot_response=response,
            ask_for_clarification=clarification_needed,
            clarification_options=None
        )
        
    except Exception as e:
        # Error handling ve analytics
        response_time = time.time() - request_start_time
        log_analytics("error", response_time, success=False, error_type=str(type(e).__name__))
        logger.error(f"Process query error: {e}", exc_info=True)
        
        # MVP: Error durumunda da usage tracking
        await get_sqlite_service().update_tenant_usage(
            tenant_id=tenant_id,
            response_time=response_time,
            energy_consumption=0.1,  # Error durumunda minimal enerji
            nlu_method="error",
            success=False
        )
        
        return NLUResponse(
            original_query=original_user_query,
            session_id=effective_session_id,
            tenant_id=tenant_id,
            nlu_method="error",
            detected_intent="error",
            actionable_message="Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.",
            bot_response="Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.",
            ask_for_clarification=False
        )

@app.get("/")
async def read_root(request: FastAPIRequest):
    api_version_message = "Chatbot NLU API (v1.3 - Final)"
    lifespan_run = getattr(request.app.state, 'lifespan_was_executed', False)
    critical_loaded = getattr(request.app.state, 'critical_resources_loaded', False)
    symspell_from_state = getattr(request.app.state, 'sym_spell', None)
    symspell_is_loaded_via_state = symspell_from_state is not None
    symspell_word_count_via_state = len(symspell_from_state.words) if symspell_is_loaded_via_state and hasattr(symspell_from_state, 'words') else 0
    
    status_detail = ""
    if lifespan_run:
        if critical_loaded and symspell_is_loaded_via_state:
            status_detail = " - Durum: Aktif, Temel NLU Kaynakları ve SymSpell Yüklendi."
            status_detail += f" SymSpell Aktif (Kelime Sayısı: {symspell_word_count_via_state})."
        else: status_detail = " - Durum: Kısmen Aktif, NLU Kaynakları Yüklendi ama SymSpell YÜKLENEMEDİ."
    else: status_detail = " - Durum: Etkin Değil, LIFESPAN ÇALIŞMADI veya erken başarısız oldu."
    
    return {
        "message": f"{api_version_message}{status_detail}",
        "app_state_lifespan_executed": lifespan_run,
        "app_state_critical_resources_loaded": critical_loaded,
        "app_state_symspell_loaded": symspell_is_loaded_via_state,
        "app_state_symspell_word_count": symspell_word_count_via_state,
        "turkish_frequency_dictionary_path_exists": TURKISH_FREQUENCY_DICTIONARY_PATH.exists(),
        "nlu_model_path_exists": MODEL_PATH.exists()
    }

def detect_intent_fast_mode(query: str) -> tuple[str, float]:
    """GÜÇLENDIRILMIŞ hızlı mod intent tespiti - 'var mı?' sorunları için optimize edilmiş"""
    query_lower = query.lower()
    
    # Önce regex kurallarını kullan (daha güvenilir)
    for intent_name, pattern in rules.items():
        if pattern.search(query_lower):
            confidence = 0.9 if intent_name == "ürün_arama" else 0.8
            return intent_name, confidence
    
    # Sonra ek pattern'ler - daha spesifik
    specific_patterns = {
        "selamlama": [
            r"^(merhaba|selam|slm|mrb|hey|günaydın|iyi günler|sa|selamun aleykum|selamlar)$"
        ],
        "ürün_arama": [
            # Ürün adı + var mı kombinasyonları
            r"\b(pijama|gecelik|sabahlık|elbise|tulum|şort|kapri|takım|abiye|tesettür|çorap|boxer|sütyen|iç çamaşırı)\s*.*(var\s+m[ıi]|varm[ıi]|var\s+mi|vr|varmi|varmı|mevcut\s+mu|bulunur\s+mu|arıyorum|lazım)\b",
            # Ürün özelliği + var mı
            r"\b(dantelli|saten|kadife|brode|dekolteli|düğmeli|askılı|kısa kollu|uzun kollu|v yaka|büyük beden)\s*.*(var\s+m[ıi]|varm[ıi]|mevcut\s+mu)\b",
            # Genel ürün arama
            r"\b(hangi ürünler|ne var|neler var|katalog|ürünleriniz|koleksiyon|modeller|çeşitler|göster|listele|ara|bul)\b"
        ],
        "stok_sorgulama": [
            # Beden + var mı
            r"\b(\d{2,3}|s|m|l|xl|xs|xxl|xxxl|small|medium|large|xlarge|numara|beden|bedeni|bedenleri)\s*.*(var\s+m[ıi]|varm[ıi]|kaldı\s+m[ıi]|mevcut\s+mu|stokta\s+m[ıi])\b",
            # Stok kelimeleri
            r"\b(stok|stokta|stokda|gelecek mi|tükendi|kaldı|mevcut|bulunur)\b"
        ],
        "renk_stok_sorgulama": [
            # Renk + herhangi bir şey + var mı
            r"\b(siyah|beyaz|kırmızı|kirmizi|mavi|yeşil|yesil|sarı|sari|pembe|mor|turuncu|kahverengi|gri|lacivert|bordo)\s*.*(var\s+m[ıi]|varm[ıi]|mevcut\s+mu|bulunur\s+mu)\b"
        ]
    }
    
    # Specific patterns'ı kontrol et
    for intent_name, patterns in specific_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                confidence = 0.9 if intent_name == "ürün_arama" else 0.8
                return intent_name, confidence
    
    # Fallback - genel pattern matching
    if "var mı" in query_lower or "varmı" in query_lower:
        # Context-aware var mı handling
        if any(word in query_lower for word in ["gecelik", "pijama", "elbise", "tulum", "takım", "çorap", "boxer"]):
            return "ürün_arama", 0.9
        elif any(word in query_lower for word in ["beden", "numara", "s", "m", "l", "xl", "stok"]):
            return "stok_sorgulama", 0.8
        elif any(word in query_lower for word in ["siyah", "beyaz", "kırmızı", "mavi", "yeşil", "sarı", "pembe", "mor"]):
            return "renk_stok_sorgulama", 0.8
        else:
            return "stok_sorgulama", 0.7  # Varsayılan
    
    # Temel intent'ler
    basic_intents = {
        "tesekkur": ["teşekkür", "sağol", "tşk", "eyw", "tamam", "ok"],
        "selamlama": ["merhaba", "selam", "mrb", "slm", "hey"],
        "fiyat_sorgulama": ["fiyat", "para", "kaç", "ücret", "maliyet"],
        "kargo_bilgisi_sorma": ["kargo", "teslimat", "gönderim", "kaç gün"],
        "iade_sorgulama": ["iade", "değişim", "geri"],
        "lokasyon_sorma": ["nerede", "adres", "konum", "mağaza"],
        "kapsam_disi": ["hava", "futbol", "şaka", "maç"]
    }
    
    for intent_name, keywords in basic_intents.items():
        if any(keyword in query_lower for keyword in keywords):
            return intent_name, 0.7
    
    return "kapsam_disi", 0.3

# Hızlı mod için gelişmiş response generation
def generate_response_fast_mode(intent: str, entities: dict, query: str, tenant_settings: dict) -> tuple[str, bool]:
    """Hızlı mod için gelişmiş cevap üretimi"""
    
    item_name = entities.get("item_name_candidate", "")
    size = entities.get("size", "")
    business_name = tenant_settings.get("business_name", "İşletmemiz")
    settings_json = tenant_settings.get("settings_json", {})
    default_responses = settings_json.get("default_responses", {})
    
    # Araştırma verilerine göre optimize edilmiş cevaplar
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
        "musteri_hizmetlerine_baglanma": "Tabii, hemen yetkiliye aktarıyorum.",
        "siparis_durumu_sorma": "Sipariş durumu hakkında bilgi veriyorum.",
        "websitesi_sorma": "Web sitemiz ve sosyal medya hesaplarımız hakkında bilgi veriyorum.",
        "ürün_arama": "Ürünlerimizi sizin için arıyorum.",
        "kategori_listesi": "Mevcut kategorilerimizi listeliyorum.",
        "kapsam_disi": "Bu konuda yardımcı olamıyorum. Size ürünlerimiz hakkında bilgi verebilirim.",
        "bilinmiyor": "Ne demek istediğinizi tam anlayamadım. Lütfen farklı bir şekilde sorabilir misiniz?"
    }
    
    response = responses.get(intent, responses["bilinmiyor"])
    clarification_needed = False
    
    # Özel intent işlemleri
    if intent == "ürün_arama":
        # Ürün arama logic'i - query'den ürün adını çıkar
        search_query = item_name if item_name else query
        
        # Query'den gereksiz kelimeleri temizle
        clean_query = search_query.lower()
        remove_words = ["göster", "listele", "ara", "bul", "var", "mı", "mi", "neler", "hangi", "ne", "takımları", "takımı"]
        for word in remove_words:
            clean_query = clean_query.replace(word, "").strip()
        
        # Boş kalırsa orijinal query'den temel kelimeleri al
        if not clean_query or len(clean_query) < 3:
            # Temel ürün kelimelerini bul
            product_keywords = ["pijama", "gecelik", "sabahlık", "elbise", "tulum", "şort", "kapri"]
            for keyword in product_keywords:
                if keyword in query.lower():
                    clean_query = keyword
                    break
            
            if not clean_query:
                clean_query = search_query
        
        products = search_products(clean_query, limit=5)
        if products:
            response = format_product_list(products)
        else:
            # Yazım hatası düzeltme dene
            corrected_query = try_spell_correction(clean_query)
            if corrected_query != clean_query:
                products = search_products(corrected_query, limit=5)
                if products:
                    response = f"'{clean_query}' yerine '{corrected_query}' aradınız mı?\n\n" + format_product_list(products)
                else:
                    response = "Aradığınız kriterlere uygun ürün bulunamadı. Farklı anahtar kelimeler deneyebilirsiniz."
            else:
                response = "Aradığınız kriterlere uygun ürün bulunamadı. Farklı anahtar kelimeler deneyebilirsiniz."
        clarification_needed = False
    elif intent == "kategori_listesi":
        # Kategori listesi
        categories = get_product_categories()
        if categories:
            response = f"Mevcut kategorilerimiz:\n" + "\n".join([f"• {cat}" for cat in categories])
        else:
            response = "Kategori listesi şu anda mevcut değil."
        clarification_needed = False
    # Netleştirme gereken durumlar
    elif intent == "bilinmiyor" and len(query.split()) < 2:
        clarification_needed = True
        response = "Hangi ürün hakkında bilgi almak istiyorsunuz?"
    elif intent == "fiyat_sorgulama":
        # Fiyat sorgulama - önce ürün arama dene
        search_query = item_name if item_name else query.replace("fiyat", "").replace("kaç", "").replace("ne kadar", "").replace("TL", "").replace("₺", "").strip()
        if search_query and len(search_query) > 2:
            products = search_products(search_query, limit=3)
            if products:
                response = f"🔍 '{search_query}' için bulunan ürünler:\n\n" + format_product_list(products)
                clarification_needed = False
            else:
                response = f"'{search_query}' için ürün bulunamadı. Ürünlerimizin fiyatları 696₺ ile 4.666₺ arasında değişmektedir. Başka bir ürün adı deneyebilirsiniz."
                clarification_needed = False
        else:
            response = "Ürünlerimizin fiyatları 696₺ ile 4.666₺ arasında değişmektedir. Hangi ürünün fiyatını öğrenmek istiyorsunuz?"
            clarification_needed = False
    elif intent == "ürün_bilgisi_sorma" and not item_name:
        # Ürün bilgisi için ürün adı yoksa, genel bilgi ver
        response = "Mağazamızda pijama takımları, gecelikler, sabahlıklar, elbiseler gibi çeşitli ürünler bulunmaktadır. Hangi ürün hakkında detaylı bilgi almak istiyorsunuz?"
    else:
        clarification_needed = False
    
    return response, clarification_needed

def load_models_lazy():
    """Modelleri lazy loading ile yükle - Hibrit sistem için"""
    global current_nlu_model, current_morphology, current_sym_spell, models_loaded
    
    if FAST_MODE:
        logger.info("⚡ FAST MODE: Ağır modeller devre dışı, sadece SQLite aktif")
        models_loaded = True
        return True
    
    logger.info("🤖 Hibrit sistem modelleri yükleniyor...")
    
    # FastText modeli
    try:
        global current_nlu_model
        model_path = BASE_DIR / "nlu_model.bin"
        if model_path.exists():
            current_nlu_model = fasttext.load_model(str(model_path))
            logger.info('LIFESPAN: FastText OK')
            print('LIFESPAN: FastText OK')
        else:
            logger.warning('LIFESPAN: FastText modeli bulunamadı')
            print('LIFESPAN: FastText modeli bulunamadı')
    except Exception as e:
        logger.error(f'LIFESPAN: FastText HATA: {e}')
        print(f'LIFESPAN: FastText HATA: {e}')
    
    # Zeyrek (sadece entity extraction için) - GEÇİCİ OLARAK DEVRE DIŞI
    logger.info("⚠️ Zeyrek geçici olarak devre dışı")
    # try:
    #     global current_morphology
    #     current_morphology = zeyrek.MorphAnalyzer()
    #     logger.info("✅ Zeyrek yüklendi")
    # except Exception as e:
    #     logger.error(f"❌ Zeyrek yüklenemedi: {e}")
    
    # SymSpell (sadece kritik kelimeler için)
    try:
        global current_sym_spell
        current_sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = BASE_DIR / "turkish_frequency_dictionary.txt"
        if dict_path.exists():
            current_sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1)
            logger.info(f"✅ SymSpell yüklendi ({len(current_sym_spell.words)} kelime)")
        else:
            logger.warning("⚠️ SymSpell sözlüğü bulunamadı")
    except Exception as e:
        logger.error(f"❌ SymSpell yüklenemedi: {e}")
    
    models_loaded = True
    logger.info("🎯 Hibrit sistem modelleri yüklendi")
    return True

# Hibrit response generation
def generate_response_hybrid(intent: str, entities: dict, query: str, tenant_settings: dict, analysis: Optional[NLUSingleAnalysis] = None) -> tuple[str, bool]:
    """Hibrit mod için gelişmiş cevap üretimi - ML + regex + context"""
    
    item_name = entities.get("item_name_candidate", "")
    size = entities.get("size", "")
    business_name = tenant_settings.get("business_name", "İşletmemiz")
    settings_json = tenant_settings.get("settings_json", {})
    default_responses = settings_json.get("default_responses", {})
    
    # ML confidence'a göre response kalitesi
    ml_confidence = analysis.confidence_score if analysis else 0.0
    
    # Özel intent işlemleri önce
    if intent == "ürün_arama":
        # Ürün arama logic'i - query'den ürün adını çıkar
        search_query = item_name if item_name else query
        
        # Query'den gereksiz kelimeleri temizle
        clean_query = search_query.lower()
        remove_words = ["göster", "listele", "ara", "bul", "var", "mı", "mi", "neler", "hangi", "ne", "takımları", "takımı"]
        for word in remove_words:
            clean_query = clean_query.replace(word, "").strip()
        
        # Boş kalırsa orijinal query'den temel kelimeleri al
        if not clean_query or len(clean_query) < 3:
            # Temel ürün kelimelerini bul
            product_keywords = ["pijama", "gecelik", "sabahlık", "elbise", "tulum", "şort", "kapri"]
            for keyword in product_keywords:
                if keyword in query.lower():
                    clean_query = keyword
                    break
            
            if not clean_query:
                clean_query = search_query
        
        products = search_products(clean_query, limit=5)
        if products:
            response = format_product_list(products)
        else:
            # Yazım hatası düzeltme dene
            corrected_query = try_spell_correction(clean_query)
            if corrected_query != clean_query:
                products = search_products(corrected_query, limit=5)
                if products:
                    response = f"'{clean_query}' yerine '{corrected_query}' aradınız mı?\n\n" + format_product_list(products)
                else:
                    response = "Aradığınız kriterlere uygun ürün bulunamadı. Farklı anahtar kelimeler deneyebilirsiniz."
            else:
                response = "Aradığınız kriterlere uygun ürün bulunamadı. Farklı anahtar kelimeler deneyebilirsiniz."
        return response, False
    elif intent == "kategori_listesi":
        # Kategori listesi
        categories = get_product_categories()
        if categories:
            response = f"Mevcut kategorilerimiz:\n" + "\n".join([f"• {cat}" for cat in categories])
        else:
            response = "Kategori listesi şu anda mevcut değil."
        return response, False
    
    # Temel responses
    base_responses = {
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
        "musteri_hizmetlerine_baglanma": "Tabii, hemen yetkiliye aktarıyorum.",
        "siparis_durumu_sorma": "Sipariş durumu hakkında bilgi veriyorum.",
        "bilinmiyor": "Ne demek istediğinizi tam anlayamadım. Lütfen farklı bir şekilde sorabilir misiniz?"
    }
    
    # ML confidence'a göre response kalitesi artır
    if ml_confidence > 0.8:
        # Yüksek güven - detaylı response
        response = base_responses.get(intent, base_responses["bilinmiyor"])
        if intent == "fiyat_sorgulama" and item_name:
            response = f"'{item_name}' ürününün fiyatı hakkında detaylı bilgi veriyorum."
        elif intent == "stok_sorgulama" and item_name:
            response = f"'{item_name}' ürününün stok durumunu kontrol ediyorum{f' ({size} beden)' if size else ''}."
    elif ml_confidence > 0.5:
        # Orta güven - standart response
        response = base_responses.get(intent, base_responses["bilinmiyor"])
    else:
        # Düşük güven - güvenli response
        response = base_responses.get(intent, base_responses["bilinmiyor"])
        if intent == "bilinmiyor":
            response = "Anladığım kadarıyla... Lütfen daha detaylı açıklayabilir misiniz?"
    
    clarification_needed = False
    
    # Netleştirme gereken durumlar
    if intent == "bilinmiyor" and len(query.split()) < 2:
        clarification_needed = True
        response = "Hangi ürün hakkında bilgi almak istiyorsunuz?"
    elif intent == "fiyat_sorgulama" and not item_name:
        # Fiyat sorgulama için ürün adı yoksa, genel fiyat bilgisi ver
        response = "Ürünlerimizin fiyatları 50 TL ile 500 TL arasında değişmektedir. Hangi ürünün fiyatını öğrenmek istiyorsunuz?"
    elif intent == "ürün_bilgisi_sorma" and not item_name:
        # Ürün bilgisi için ürün adı yoksa, genel bilgi ver
        response = "Mağazamızda pantolon, gömlek, elbise, ceket gibi çeşitli ürünler bulunmaktadır. Hangi ürün hakkında detaylı bilgi almak istiyorsunuz?"
    else:
        clarification_needed = False
    
    return response, clarification_needed

# MVP için monitoring ve analytics
request_analytics = {
    "total_requests": 0,
    "intent_counts": defaultdict(int),
    "response_times": [],
    "error_counts": defaultdict(int),
    "session_counts": defaultdict(int),
    "start_time": datetime.now()
}

# Session Cache - Son konuşmaları hafızada tut
session_cache = {}
MAX_CACHE_SIZE = 1000
CACHE_EXPIRY_MINUTES = 30

def get_session_context(session_id: str, tenant_id: int) -> dict:
    """Session context'ini getir"""
    cache_key = f"{tenant_id}_{session_id}"
    now = datetime.now()
    
    if cache_key in session_cache:
        session_data = session_cache[cache_key]
        # Expiry kontrol
        if (now - session_data['last_updated']).total_seconds() < CACHE_EXPIRY_MINUTES * 60:
            return session_data.get('context', {})
    
    return {}

def update_session_context(session_id: str, tenant_id: int, query: str, intent: str, response: str):
    """Session context'ini güncelle"""
    cache_key = f"{tenant_id}_{session_id}"
    now = datetime.now()
    
    # Cache boyut kontrolü
    if len(session_cache) > MAX_CACHE_SIZE:
        # En eski 100 kaydı sil
        oldest_keys = sorted(session_cache.keys(), 
                           key=lambda k: session_cache[k]['last_updated'])[:100]
        for key in oldest_keys:
            del session_cache[key]
    
    if cache_key not in session_cache:
        session_cache[cache_key] = {
            'context': {
                'last_queries': [],
                'last_intents': [],
                'last_products_searched': [],
                'preferred_category': None
            },
            'last_updated': now
        }
    
    context = session_cache[cache_key]['context']
    
    # Son sorguları sakla (max 5)
    context['last_queries'].append(query)
    if len(context['last_queries']) > 5:
        context['last_queries'] = context['last_queries'][-5:]
    
    # Son intent'leri sakla
    context['last_intents'].append(intent)
    if len(context['last_intents']) > 5:
        context['last_intents'] = context['last_intents'][-5:]
    
    # Ürün arama geçmişi
    if intent == "ürün_arama" and query:
        context['last_products_searched'].append(query)
        if len(context['last_products_searched']) > 10:
            context['last_products_searched'] = context['last_products_searched'][-10:]
    
    # Tercih edilen kategori analizi
    if intent == "ürün_arama":
        category_hints = {
            'pijama': 'Pijama Takımları',
            'gecelik': 'Gecelikler', 
            'sabahlık': 'Sabahlıklar',
            'elbise': 'Elbiseler'
        }
        for hint, category in category_hints.items():
            if hint in query.lower():
                context['preferred_category'] = category
                break
    
    session_cache[cache_key]['last_updated'] = now

# MVP Rate Limiting
tenant_rate_limits = defaultdict(lambda: {"count": 0, "reset_time": datetime.now()})

def check_rate_limit(tenant_id: int) -> tuple[bool, str]:
    """Tenant için rate limit kontrolü"""
    now = datetime.now()
    tenant_limit = tenant_rate_limits[tenant_id]
    
    # Günlük reset kontrolü
    if (now - tenant_limit["reset_time"]).days >= 1:
        tenant_limit["count"] = 0
        tenant_limit["reset_time"] = now
    
    # Limit kontrolü (varsayılan: günde 1000 query)
    daily_limit = 1000  # Bu değer tenant ayarlarından alınabilir
    
    if tenant_limit["count"] >= daily_limit:
        return False, f"Günlük query limiti aşıldı ({daily_limit})"
    
    tenant_limit["count"] += 1
    return True, "OK"

def log_analytics(intent: str, response_time: float, success: bool = True, error_type: str = None):
    """Analytics verilerini logla"""
    request_analytics["total_requests"] += 1
    request_analytics["intent_counts"][intent] += 1
    request_analytics["response_times"].append(response_time)
    
    if not success and error_type:
        request_analytics["error_counts"][error_type] += 1
    
    # Son 1000 isteği tut
    if len(request_analytics["response_times"]) > 1000:
        request_analytics["response_times"] = request_analytics["response_times"][-1000:]

def get_analytics_summary():
    """Analytics özetini döndür"""
    if not request_analytics["response_times"]:
        return {"message": "Henüz veri yok"}
    
    avg_response_time = sum(request_analytics["response_times"]) / len(request_analytics["response_times"])
    uptime = datetime.now() - request_analytics["start_time"]
    
    return {
        "total_requests": request_analytics["total_requests"],
        "avg_response_time_ms": round(avg_response_time * 1000, 2),
        "uptime_seconds": int(uptime.total_seconds()),
        "top_intents": dict(sorted(request_analytics["intent_counts"].items(), key=lambda x: x[1], reverse=True)[:5]),
        "error_counts": dict(request_analytics["error_counts"]),
        "requests_per_minute": round(request_analytics["total_requests"] / (uptime.total_seconds() / 60), 2)
    }

@app.get("/analytics")
async def get_analytics():
    """MVP Analytics - Bot performans metrikleri"""
    return get_analytics_summary()

@app.get("/tenant/{tenant_id}/usage")
async def get_tenant_usage(tenant_id: int, days: int = 30):
    """MVP Tenant Usage - Belirli tenant'ın kullanım istatistikleri"""
    try:
        usage_data = await get_sqlite_service().get_tenant_usage(tenant_id, days)
        if not usage_data:
            raise HTTPException(status_code=404, detail="Tenant bulunamadı")
        
        return {
            "status": "success",
            "data": usage_data,
            "message": f"Tenant {tenant_id} kullanım istatistikleri"
        }
    except Exception as e:
        logger.error(f"Get tenant usage error: {e}")
        raise HTTPException(status_code=500, detail="Kullanım verileri alınamadı")

@app.get("/tenant/{tenant_id}/billing")
async def get_tenant_billing(tenant_id: int):
    """MVP Tenant Billing - Belirli tenant'ın fatura özeti"""
    try:
        billing_data = await get_sqlite_service().get_tenant_billing(tenant_id)
        if not billing_data:
            raise HTTPException(status_code=404, detail="Tenant bulunamadı")
        
        return {
            "status": "success",
            "data": billing_data,
            "message": f"Tenant {tenant_id} fatura özeti"
        }
    except Exception as e:
        logger.error(f"Get tenant billing error: {e}")
        raise HTTPException(status_code=500, detail="Fatura verileri alınamadı")

@app.get("/tenants/usage/summary")
async def get_tenants_usage_summary():
    """MVP Tenants Summary - Tüm tenant'ların kullanım özeti"""
    try:
        summary_data = await get_sqlite_service().get_tenants_usage_summary()
        return {
            "status": "success",
            "data": summary_data,
            "message": "Tüm tenant'ların kullanım özeti"
        }
    except Exception as e:
        logger.error(f"Get tenants summary error: {e}")
        raise HTTPException(status_code=500, detail="Özet verileri alınamadı")

@app.get("/health")
async def health_check():
    """Health check endpoint - MVP için"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "sqlite": "active",
            "fasttext": "active" if current_nlu_model else "inactive",
            "zeyrek": "active" if current_morphology else "inactive",
            "symspell": "active" if current_sym_spell else "inactive"
        }
    }

@app.post("/clear-cache")
async def clear_cache():
    """Cache temizleme endpoint"""
    global query_cache
    query_cache.clear()
    return {"status": "success", "message": "Cache temizlendi"}

# Ürün Yönetimi API Endpoints
@app.get("/products")
async def get_products(limit: int = 50):
    """Ürünleri listele"""
    try:
        conn = sqlite3.connect('chatbot_data.db')
        cursor = conn.cursor()
        
        # İstatistikler
        cursor.execute('SELECT COUNT(*) FROM products WHERE is_active = 1')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT category) FROM products WHERE is_active = 1')
        categories = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(original_price) FROM products WHERE is_active = 1 AND original_price IS NOT NULL')
        avg_price = cursor.fetchone()[0] or 0
        
        # Ürünler
        cursor.execute('''
            SELECT product_code, product_name, color, original_price, discounted_price, 
                   discount_rate, category, description
            FROM products 
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        products = []
        for row in cursor.fetchall():
            products.append({
                "product_code": row[0],
                "product_name": row[1],
                "color": row[2],
                "original_price": row[3],
                "discounted_price": row[4],
                "discount_rate": row[5],
                "category": row[6],
                "description": row[7]
            })
        
        conn.close()
        
        return {
            "total": total,
            "categories": categories,
            "avg_price": round(avg_price, 2),
            "products": products
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/search")
async def search_products_api(q: str, limit: int = 20):
    """Ürün arama"""
    try:
        products = search_products(q, limit=limit)
        
        result = []
        for product in products:
            result.append({
                "product_code": product[0],
                "product_name": product[1],
                "color": product[2],
                "original_price": product[3],
                "discounted_price": product[4],
                "discount_rate": product[5],
                "category": product[6],
                "description": product[7]
            })
        
        return {"products": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/products")
async def add_product(product: dict):
    """Yeni ürün ekle"""
    try:
        conn = sqlite3.connect('chatbot_data.db')
        cursor = conn.cursor()
        
        # İndirimli fiyat hesapla
        original_price = product.get('original_price')
        discount_rate = product.get('discount_rate', 0)
        discounted_price = original_price * (1 - discount_rate / 100) if discount_rate > 0 else None
        
        # Açıklama oluştur
        description = create_description(
            product.get('product_name') or 'Ürün',
            product.get('color') or 'Renk Belirtilmemiş',
            original_price or 0.0,
            discounted_price,
            discount_rate
        )
        
        cursor.execute('''
            INSERT INTO products 
            (product_code, product_name, color, original_price, discount_rate, 
             discounted_price, category, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            product.get('product_code'),
            product.get('product_name'),
            product.get('color'),
            original_price,
            discount_rate,
            discounted_price,
            product.get('category'),
            description
        ))
        
        conn.commit()
        conn.close()
        
        return {"message": "Ürün başarıyla eklendi"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/products/{product_code}")
async def delete_product(product_code: str):
    """Ürün sil"""
    try:
        conn = sqlite3.connect('chatbot_data.db')
        cursor = conn.cursor()
        
        cursor.execute('UPDATE products SET is_active = 0 WHERE product_code = ?', (product_code,))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Ürün bulunamadı")
        
        conn.commit()
        conn.close()
        
        return {"message": "Ürün başarıyla silindi"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("İşlem zaman aşımına uğradı")

def run_with_timeout(func, timeout_seconds=10):
    """Fonksiyonu timeout ile çalıştır"""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                result = future.result(timeout=timeout_seconds)
                return result, None
            except concurrent.futures.TimeoutError:
                future.cancel()
                return None, "timeout"
    except Exception as e:
        return None, str(e)

def resolve_intent_conflict(query: str, ml_analysis, regex_intent: str, regex_confidence: float) -> str:
    """Context-aware intent conflict resolution with PRODUCT SEARCH PRIORITY"""
    query_lower = query.lower()
    
    # ÖNCELİK 1: ÜRÜN ARAMA - EN YÜKSEK ÖNCELİK (ML'yi override et)
    product_keywords = ["pijama", "gecelik", "sabahlık", "elbise", "takım", "abiye", "tesettür", "günlük elbise", "kışlık mont", "keten pantolon", "pamuklu gömlek"]
    search_keywords = ["var mı", "varmı", "var mi", "vr", "mevcut mu", "bulunur mu", "satıyor musunuz", "arıyorum", "lazım", "göster", "ara", "bul"]
    
    # Ürün kelimesi + arama kelimesi kombinasyonu - Daha sıkı kontrol
    has_product = any(word in query_lower for word in product_keywords)
    has_search = any(word in query_lower for word in search_keywords)
    
    if has_product and has_search:
        return "ürün_arama"
    
    # ÇOK ÖNEMLİ: Özel ürün arama pattern'leri - ML'yi tamamen override et
    if re.search(r"\b(gecelik|pijama|sabahlık|elbise|takım|abiye|tesettür|keten\s+pantolon|pamuklu\s+gömlek)\s+.*(var\s+m[ıi]|varm[ıi]|mevcut\s+mu|bulunur\s+mu|satıyor\s+musunuz|arıyorum|lazım)\b", query_lower):
        return "ürün_arama"
    
    # İkinci pattern - malzeme + ürün + var mı
    if re.search(r"\b(keten|pamuk|ipek|deri)\s+(pantolon|gömlek|elbise|ceket)\s+.*(var\s+m[ıi]|varm[ıi]|mevcut\s+mu)\b", query_lower):
        return "ürün_arama"
    
    # ÖNCELİK 2: Kategori listesi
    if any(word in query_lower for word in ["kategori", "kategoriler", "çeşit", "türler", "neler var"]):
        return "kategori_listesi"
    
    # ÖNCELİK 3: Regex sonucu ürün arama ise
    if regex_intent == "ürün_arama":
        return "ürün_arama"
    
    # ÖNCELİK 4: "var mı" çakışması çözümü - CONTEXT-AWARE
    if "var mı" in query_lower or "varmı" in query_lower:
        # Context'e göre karar ver
        if any(word in query_lower for word in ["iade", "değişim", "değiştir", "geri"]):
            return "iade_sorgulama"
        elif any(word in query_lower for word in ["web", "site", "instagram", "facebook", "link"]):
            return "websitesi_sorma"
        elif any(word in query_lower for word in ["renk", "renkler", "başka renk", "farklı renk"]):
            return "ürün_bilgisi_sorma"
        elif any(word in query_lower for word in ["mağaza", "adres", "nerede", "konum", "yer"]):
            return "lokasyon_sorma"
        elif any(word in query_lower for word in ["indirim", "kampanya", "kod"]):
            return "indirim_kampanya_sorma"
        elif any(word in query_lower for word in ["beden", "numara", "s", "m", "l", "xl", "stok"]):
            return "stok_sorgulama"
        elif has_product:  # Ürün kelimesi varsa ürün arama
            return "ürün_arama"
        else:
            # Varsayılan olarak stok_sorgulama
            return "stok_sorgulama"
    
    # ÖNCELİK 5: Çoklu soru tespiti
    question_marks = query.count("?")
    if question_marks > 1:
        first_question = query.split("?")[0] + "?"
        if "beden" in first_question.lower() or "stok" in first_question.lower():
            return "stok_sorgulama"
        elif "fiyat" in first_question.lower() or "kaç" in first_question.lower():
            return "fiyat_sorgulama"
        elif "kargo" in first_question.lower():
            return "kargo_bilgisi_sorma"
    
    # ÖNCELİK 6: Lokasyon sorma için özel kontrol
    if any(word in query_lower for word in ["mağaza", "adres", "nerede", "konum", "yer", "lokasyon"]):
        return "lokasyon_sorma"
    
    # ÖNCELİK 7: Regex vs ML karar verme
    if regex_intent in ["ürün_arama", "kategori_listesi"] and regex_confidence > 0.3:
        return regex_intent
    elif ml_analysis and ml_analysis.confidence_score > 0.7:
        return ml_analysis.slm_intent
    elif regex_confidence > 0.5:
        return regex_intent
    elif ml_analysis and ml_analysis.confidence_score > 0.4:
        return ml_analysis.slm_intent
    else:
        return "kapsam_disi"

# Ürün veritabanı fonksiyonları
def search_products(query: str, category: str = None, max_price: float = None, limit: int = 10):
    """Ürün arama fonksiyonu"""
    conn = sqlite3.connect('chatbot_data.db')
    cursor = conn.cursor()
    
    # Temel SQL sorgusu
    sql = """
        SELECT product_code, product_name, color, original_price, discounted_price, 
               discount_rate, category, description
        FROM products 
        WHERE is_active = 1
    """
    params = []
    
    # Arama kriteri ekle
    if query:
        sql += " AND (LOWER(product_name) LIKE LOWER(?) OR LOWER(description) LIKE LOWER(?) OR LOWER(color) LIKE LOWER(?))"
        search_term = f"%{query}%"
        params.extend([search_term, search_term, search_term])
    
    # Kategori filtresi
    if category:
        sql += " AND category = ?"
        params.append(category)
    
    # Fiyat filtresi
    if max_price:
        sql += " AND (discounted_price <= ? OR (discounted_price IS NULL AND original_price <= ?))"
        params.extend([max_price, max_price])
    
    # Sıralama ve limit
    sql += " ORDER BY discounted_price ASC, original_price ASC LIMIT ?"
    params.append(limit)
    
    cursor.execute(sql, params)
    results = cursor.fetchall()
    conn.close()
    
    return results

def get_product_categories():
    """Mevcut kategorileri getir"""
    conn = sqlite3.connect('chatbot_data.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT category FROM products WHERE is_active = 1 ORDER BY category")
    categories = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return categories

def get_product_by_code(product_code: str):
    """Ürün koduna göre ürün getir"""
    conn = sqlite3.connect('chatbot_data.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT product_code, product_name, color, original_price, discounted_price, 
               discount_rate, category, description
        FROM products 
        WHERE product_code = ? AND is_active = 1
    """, (product_code,))
    
    result = cursor.fetchone()
    conn.close()
    
    return result

def format_product_info(product_data):
    """Ürün bilgisini formatla"""
    if not product_data:
        return "Ürün bulunamadı."
    
    code, name, color, original_price, discounted_price, discount_rate, category, description = product_data
    
    info = f"🏷️ **{name}**\n"
    info += f"📦 Kod: {code}\n"
    info += f"🎨 Renk: {color}\n"
    info += f"📂 Kategori: {category}\n"
    
    # Fiyat bilgisi
    if discounted_price and original_price:
        savings = original_price - discounted_price
        info += f"💰 Fiyat: ~~{original_price:.0f}₺~~ **{discounted_price:.0f}₺**\n"
        info += f"💸 Tasarruf: {savings:.0f}₺ (%{discount_rate:.0f} indirim)\n"
    elif original_price:
        info += f"💰 Fiyat: {original_price:.0f}₺\n"
    
    if description:
        info += f"ℹ️ Özellikler: {description}\n"
    
    return info

def format_product_list(products):
    """Ürün listesini formatla"""
    if not products:
        return "❌ Aradığınız kriterlere uygun ürün bulunamadı."
    
    result = f"🛍️ **{len(products)} ürün bulundu:**\n\n"
    
    for i, product in enumerate(products, 1):
        code, name, color, original_price, discounted_price, discount_rate, category, description = product
        
        result += f"**{i}. {name[:50]}{'...' if len(name) > 50 else ''}**\n"
        result += f"   📦 {code} | 🎨 {color} | 📂 {category}\n"
        
        # Fiyat bilgisi
        if discounted_price and original_price:
            result += f"   💰 ~~{original_price:.0f}₺~~ **{discounted_price:.0f}₺** (%{discount_rate:.0f} indirim)\n"
        elif original_price:
            result += f"   💰 {original_price:.0f}₺\n"
        
        result += "\n"
    
    return result

# Eski try_spell_correction fonksiyonu kaldırıldı - Yeni ColorMapper sistemi kullanılıyor

def create_description(product_name: str, color: str, original_price: float, discounted_price: float = None, discount_rate: float = 0) -> str:
    """Ürün için açıklama oluştur"""
    if not product_name:
        return ""
    
    description = f"{product_name}"
    
    if color:
        description += f" - {color} renk"
    
    if original_price:
        description += f" - {original_price:.0f}₺"
        
        if discounted_price and discount_rate > 0:
            description += f" (İndirimli: {discounted_price:.0f}₺, %{discount_rate:.0f} indirim)"
    
    return description

# Query cache - Sık sorulan sorular için cache
query_cache = {}
MAX_QUERY_CACHE_SIZE = 500

def get_cached_intent(query: str) -> Optional[tuple]:
    """Cache'den intent al"""
    normalized_query = query.lower().strip()
    if normalized_query in query_cache:
        return query_cache[normalized_query]
    return None

def cache_intent_result(query: str, intent: str, confidence: float, method: str):
    """Intent sonucunu cache'le"""
    normalized_query = query.lower().strip()
    
    # Cache boyut kontrolü
    if len(query_cache) > MAX_QUERY_CACHE_SIZE:
        # En eski 100 kaydı sil
        oldest_keys = list(query_cache.keys())[:100]
        for key in oldest_keys:
            del query_cache[key]
    
    query_cache[normalized_query] = (intent, confidence, method)

# RENK MAP SINIFI - Daha verimli renk eşleştirme sistemi
class ColorMapper:
    """Efficient color mapping with Turkish normalization and caching"""
    
    def __init__(self):
        # Ana renk haritası - Türkçe normalizasyon
        self._color_map = {
            # Temel renkler
            "siyah": "SİYAH", "siyhi": "SİYAH", "siyaj": "SİYAH", "siyha": "SİYAH",
            "beyaz": "BEYAZ", "byaz": "BEYAZ", "beyza": "BEYAZ", "byez": "BEYAZ",
            "kirmizi": "KIRMIZI", "kırmızı": "KIRMIZI", "krmzi": "KIRMIZI", "krmızı": "KIRMIZI",
            "mavi": "MAVİ", "mvi": "MAVİ", "mvai": "MAVİ", "mavı": "MAVİ",
            "yeşil": "YEŞİL", "yesil": "YEŞİL", "yesl": "YEŞİL", "yşl": "YEŞİL",
            "sarı": "SARI", "sari": "SARI", "sar": "SARI", "sarj": "SARI",
            "pembe": "PEMBE", "pmbe": "PEMBE", "penbe": "PEMBE",
            "mor": "MOR", "mr": "MOR", "mour": "MOR",
            "turuncu": "TURUNCU", "trunc": "TURUNCU", "turunc": "TURUNCU",
            "kahverengi": "KAHVERENGİ", "khvrengi": "KAHVERENGİ", "kahvrengi": "KAHVERENGİ",
            "gri": "GRİ", "gr": "GRİ", "gry": "GRİ",
            "lacivert": "LACİVERT", "lacvrt": "LACİVERT", "lacivet": "LACİVERT",
            "bordo": "BORDO", "brdo": "BORDO", "bord": "BORDO",
            # Ek varyasyonlar
            "siyahı": "SİYAH", "beyazı": "BEYAZ", "kırmızısı": "KIRMIZI",
            "mavisi": "MAVİ", "yeşili": "YEŞİL", "sarısı": "SARI",
            "pembesi": "PEMBE", "moru": "MOR", "turuncusu": "TURUNCU",
            "grisi": "GRİ", "laciverti": "LACİVERT", "bordosu": "BORDO"
        }
        
        # Cache için
        self._cache = {}
        self._max_cache_size = 200
    
    def normalize_color(self, text: str) -> str:
        """Renk normalize etme - cache'li ve optimize edilmiş"""
        if not text or len(text) < 2:
            return text
            
        # Cache kontrolü
        text_lower = text.lower().strip()
        if text_lower in self._cache:
            return self._cache[text_lower]
        
        # Direkt mapping kontrolü
        if text_lower in self._color_map:
            result = text.replace(text_lower, self._color_map[text_lower])
        else:
            # Fuzzy matching - sadece renk kelimeleri içinse
            result = text
            for turkish_color, standard_color in self._color_map.items():
                if turkish_color in text_lower:
                    result = text.replace(turkish_color, standard_color)
                    break
        
        # Cache'e ekle (boyut kontrolü ile)
        if len(self._cache) > self._max_cache_size:
            # En eski 50 kaydı sil
            oldest_keys = list(self._cache.keys())[:50]
            for key in oldest_keys:
                del self._cache[key]
        
        self._cache[text_lower] = result
        return result
    
    def extract_colors(self, text: str) -> list:
        """Metinden renkleri çıkar"""
        found_colors = []
        text_lower = text.lower()
        
        for turkish_color, standard_color in self._color_map.items():
            if turkish_color in text_lower:
                if standard_color not in found_colors:
                    found_colors.append(standard_color)
        
        return found_colors

# Global color mapper instance
color_mapper = ColorMapper()

# OPTİMİZE EDİLMİŞ YAZIM DÜZELTMESİ
def try_spell_correction(query: str) -> str:
    """Optimized spell correction with color normalization"""
    if not query or len(query) < 3:
        return query
    
    # Renk normalizasyonu (yeni sistem)
    corrected_query = color_mapper.normalize_color(query)
    
    # Temel ürün yazım hataları (sınırlı liste)
    basic_corrections = {
        "pijma": "pijama", "pjama": "pijama", "piyama": "pijama",
        "geclik": "gecelik", "gecelik": "gecelik",
        "sabahlik": "sabahlık", "sabalik": "sabahlık",
        "elbse": "elbise", "elise": "elbise",
        "tulm": "tulum", "danteli": "dantelli",
        "kusak": "kuşaklı", "kuşak": "kuşaklı"
    }
    
    query_lower = corrected_query.lower()
    for wrong, correct in basic_corrections.items():
        if wrong in query_lower:
            corrected_query = corrected_query.replace(wrong, correct)
            break  # İlk match'te dur (performans için)
    
    return corrected_query

if __name__ == "__main__":
    import uvicorn
    print("🚀 Chatbot NLU API Başlatılıyor...")
    print(f"📊 FAST_MODE: {FAST_MODE}")
    print(f"🌐 Server: http://localhost:8000")
    print(f"📚 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
