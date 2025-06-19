# database_service.py
import logging
from typing import Optional, List, Dict, Any
from supabase import Client as SupabaseClient
from datetime import datetime, timezone
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

async def get_tenant_settings(supabase: SupabaseClient, tenant_id: int) -> Optional[dict]:
    if not supabase:
        logger.error("Supabase client sağlanmadı (get_tenant_settings).")
        return None
    try:
        response = supabase.table("tenants")\
            .select("id, business_name, business_type, settings_json")\
            .eq("id", tenant_id)\
            .limit(1)\
            .execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Tenant {tenant_id} ayarları çekilirken istisna: {e}", exc_info=True)
        return None

async def get_all_products_by_category(supabase: SupabaseClient, tenant_id: int, business_type: str, category: str) -> List[Dict[str, Any]]:
    """
    Belirli bir kategoriye ait tüm ürünleri çeker.
    Kategori adları için büyük/küçük harf duyarsız arama (ilike) yapar.
    """
    if not supabase or not category: return []
    
    target_table = ""
    if business_type.strip().lower() == "giyim":
        target_table = "clothing_products"
    # Diğer iş türleri için elif blokları eklenebilir.
    else:
        return []

    try:
        logger.info(f"--- KATEGORİ ARAMASI: '{target_table}' tablosunda kategori '{category}' aranıyor...")
        response = supabase.table(target_table)\
            .select("id, name")\
            .eq("tenant_id", tenant_id)\
            .ilike("category", f"%{category}%")\
            .execute()
        
        if response.data:
            logger.info(f"{len(response.data)} adet ürün '{category}' kategorisinde bulundu.")
            return response.data
        return []
    except Exception as e:
        logger.error(f"Kategori '{category}' için ürün aranırken hata: {e}", exc_info=True)
        return []


async def get_items_by_name_fuzzy(supabase: SupabaseClient, tenant_id: int, business_type: str, item_name_candidate: str, limit: int = 5) -> List[Dict[str, Any]]:
    if not supabase: return []
    
    target_table = ""
    normalized_business_type = business_type.strip().lower()
    if normalized_business_type == "giyim":
        target_table = "clothing_products"
    else:
        logger.warning(f"Desteklenmeyen işletme türü '{business_type}' için öğe tablosu bilinmiyor.")
        return []

    try:
        # Eğer item_name_candidate boşsa, tüm ürünleri getir
        if not item_name_candidate or not item_name_candidate.strip():
            all_products_response = supabase.table(target_table)\
                .select("id, name, price, category, attributes_json")\
                .eq("tenant_id", tenant_id)\
                .limit(limit)\
                .execute()
            
            if all_products_response.data:
                logger.info(f"Boş arama için {len(all_products_response.data)} ürün getirildi.")
                return all_products_response.data
            return []

        all_products_response = supabase.table(target_table)\
            .select("id, name, price, category, attributes_json")\
            .eq("tenant_id", tenant_id)\
            .execute()

        if not all_products_response.data: return []
            
        all_products = all_products_response.data
        product_names = [product['name'] for product in all_products]
        product_map = {product['name']: product for product in all_products}
        best_matches = process.extract(item_name_candidate, product_names, scorer=fuzz.WRatio, limit=limit, score_cutoff=60)

        if not best_matches: return []
        
        return [product_map[match[0]] for match in best_matches]

    except Exception as e:
        logger.error(f"Tenant {tenant_id}, item '{item_name_candidate}' için akıllı arama sırasında hata: {e}", exc_info=True)
        return []

async def get_item_by_id(supabase: SupabaseClient, tenant_id: int, business_type: str, item_id: Any) -> Optional[Dict[str, Any]]:
    if not supabase: return None
    target_table = ""
    normalized_business_type = business_type.strip().lower()
    if normalized_business_type == "giyim": target_table = "clothing_products"
    else: return None
    try:
        response = supabase.table(target_table)\
            .select("id, name, price, category, attributes_json")\
            .eq("tenant_id", tenant_id)\
            .eq("id", item_id)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data and len(response.data) > 0 else None
    except Exception as e:
        logger.error(f"Tenant {tenant_id}, ID {item_id} ile öğe çekilirken hata: {e}", exc_info=True)
        return None

async def log_conversation_turn(supabase: SupabaseClient, log_data: dict) -> bool:
    if not supabase: return False
    try:
        supabase.table("conversation_logs").insert(log_data).execute()
        return True
    except Exception as e:
        logger.error(f"Konuşma adımı loglanırken istisna: {e}", exc_info=True)
        return False

async def get_session_data(supabase: SupabaseClient, session_id: str, tenant_id: int) -> Optional[dict]:
    if not supabase: return None
    try:
        response = supabase.table("sessions")\
            .select("context_data")\
            .eq("session_id", session_id)\
            .eq("tenant_id", tenant_id)\
            .limit(1)\
            .execute()
        return response.data[0].get("context_data", {}) if response.data and len(response.data) > 0 else {}
    except Exception as e:
        logger.error(f"Oturum ({session_id}) verisi çekilirken hata: {e}", exc_info=True)
        return None

async def save_session_data(supabase: SupabaseClient, session_id: str, tenant_id: int, data: dict):
    if not supabase: return False
    try:
        supabase.table("sessions").upsert({
            "session_id": session_id,
            "tenant_id": tenant_id,
            "context_data": data,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Oturum ({session_id}) verisi kaydedilirken istisna: {e}", exc_info=True)
        return False
