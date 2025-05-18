from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import re
import fasttext
from pathlib import Path
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# Simüle edilmiş işletme verileri (Supabase şemasına uygun alanlar eklendi)
simulated_business_data = {
    "business_info": {
        "name": "Mantarinyo Butik",
        "phone": "0555 123 4567",
        "address": "Moda Caddesi No:1 Kadıköy, İstanbul",
        "website": "http://www.mantarinyo-butik.com",
        "email": "info@mantarinyo-butik.com",
        "maps_link": "https://maps.app.goo.gl/örnekLink", # Örnek bir harita linki
        "whatsapp_number": "0555 123 4567", # Supabase şemasında vardı
        "shipping_info": "Siparişleriniz Yurtiçi Kargo ile 2-3 iş gününde teslim edilir. Kargo ücreti 50 TL'dir, 500 TL ve üzeri alışverişlerde kargo ücretsizdir.",
        "return_policy": "Ürünleri teslim aldıktan sonra 14 gün içinde, kullanılmamış ve etiketi üzerinde olmak kaydıyla faturanızla birlikte iade edebilirsiniz. İade kargo ücreti müşteriye aittir.",
        "opening_hours": "Haftaiçi: 09:00-19:00, Cumartesi: 10:00-18:00, Pazar: Kapalı",
        "payment_options": "Kredi Kartı (Visa, Mastercard, Amex), Banka Kartı, Havale/EFT kabul edilmektedir."
    },
    "products": {
        "keten pantolon": {
            "link": "http://www.mantarinyo-butik.com/keten-pantolon",
            "price": "850 TL", # Örnek bir fiyat
            "description": "Yüksek kaliteli, %100 doğal ketenden üretilmiş, yaz ayları için ideal, rahat kesim pantolon.",
            "available_sizes_info": "S, M, L, XL ve 38-46 arası bedenler genellikle bulunur. Detaylar ürün sayfasında.",
            "material_composition": "%100 Keten"
        },
        "ipek gömlek": {
            "link": "http://www.mantarinyo-butik.com/ipek-gomlek",
            "price": "1250 TL", # Örnek bir fiyat
            "description": "%100 saf ipekten üretilmiş, özel günler ve günlük kullanım için şık gömlek.",
            "available_sizes_info": "S, M, L bedenleri mevcuttur. Beden tablosu için linke tıklayın.",
            "material_composition": "%100 İpek"
        },
        "deri ceket": {
            "link": "http://www.mantarinyo-butik.com/deri-ceket",
            "price": "3200 TL", # Örnek bir fiyat
            "description": "Hakiki kuzu derisinden, klasik motorcu kesim, uzun ömürlü ceket.",
            "available_sizes_info": "M, L, XL, XXL. Detaylar ürün sayfasında.",
            "material_composition": "Hakiki Kuzu Derisi, İç Astar: Pamuk"
        }
        # ... diğer ürünler eklenebilir
    }
}

app = FastAPI()

# --- Pydantic Modelleri ---
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class NLUSingleAnalysis(BaseModel):
    slm_intent: str
    slm_entities: List[Dict[str, Any]]
    confidence_score: float
    message: str

class NLUResponse(BaseModel):
    original_query: str
    session_id: str
    nlu_method: str
    analysis: Optional[NLUSingleAnalysis] = None
    detected_intent: Optional[str] = None
    previous_query_in_session: Optional[str] = None
    resolved_product: Optional[str] = None
    resolved_size: Optional[str] = None
    actionable_message: Optional[str] = None
    bot_response: Optional[str] = None

# --- OTURUM VE BAĞLAM YÖNETİMİ ---
active_sessions: Dict[str, Dict[str, Any]] = {}
SESSION_TIMEOUT_MINUTES = 60 # İleride kullanılacak

# --- MODEL YÜKLEME ---
MODEL_PATH = Path("nlu_model.bin")
nlu_model = None

@app.on_event("startup")
async def load_model():
    global nlu_model
    if MODEL_PATH.exists():
        try:
            nlu_model = fasttext.load_model(str(MODEL_PATH))
            print(f"NLU modeli '{MODEL_PATH}' başarıyla yüklendi.")
        except Exception as e:
            print(f"HATA: NLU modeli '{MODEL_PATH}' yüklenirken bir sorun oluştu: {e}")
            nlu_model = None
    else:
        print(f"HATA: NLU modeli '{MODEL_PATH}' bulunamadı. Lütfen önce 'train_model.py' betiğini çalıştırarak modeli eğitin.")

rules = {
    "fiyat_sorgulama": re.compile(r"(ne kadar|kaç para|fiyatı ne|ücreti nedir|yıllık ne kadar)", re.IGNORECASE),
    "selamlama": re.compile(r"(merhaba|selam|iyi günler|günaydın)", re.IGNORECASE),
    "iade_sorgulama": re.compile(r"(iade|geri verme|değişim|iade edebilir miyim|iade koşulları)", re.IGNORECASE),
    "stok_sorgulama": re.compile(r"(\b\w+\s+(bedeni|numarası|rengi|modeli)?\s*var\s*mı|\bstokta mevcut mu|bulunur mu|kaldı mı)", re.IGNORECASE)
}

def extract_simple_entities(query: str, intent: Optional[str] = None) -> Dict[str, Optional[str]]:
    entities = {"product": None, "size": None}
    original_query = query # Ürün aramak için orijinal sorguyu koru
    processed_query_for_analysis = query.lower() # Diğer analizler için küçük harf

    # Beden çıkarımı
    size_match = re.search(r"\b(\d{2,3}|[xsmlXSML]+(\s*beden)?|small|medium|large|ekstra large)\b", processed_query_for_analysis)
    if size_match:
        entities["size"] = size_match.group(1).upper().replace(" BEDEN", "")
        # Beden bilgisini, ürün adı arayacağımız sorgudan (hem orijinal hem de işlenmiş) çıkaralım
        processed_query_for_analysis = processed_query_for_analysis.replace(size_match.group(0).lower(), "").strip()
        # Orijinal sorgudan da çıkarırken büyük/küçük harf duyarsız olalım
        original_query_for_product_search = re.sub(re.escape(size_match.group(0)), "", original_query, flags=re.IGNORECASE).strip()
    else:
        original_query_for_product_search = original_query # Beden bulunamadıysa orijinal sorgu aynı kalsın


    product_candidate = None

    # Kural 1: Sorgu çok kısaysa veya sadece beden ve "var mı/kaldı mı" gibi bir şeyse, ürün adı yoktur.
    remaining_words_after_size_extraction = processed_query_for_analysis.split()
    if len(remaining_words_after_size_extraction) <= 2 and \
       any(kw in processed_query_for_analysis for kw in ["var mı", "kaldı mı", "mevcut mu", "peki", "si", "mı", "mu"]):
        entities["product"] = None # Kesinlikle ürün yok, bağlama güvenilecek
        print(f"DEBUG (extract_simple_entities - Kural 1): Query='{query}', Intent='{intent}' -> Entities={entities}")
        return entities

    # Kural 2: Sorgu "bu", "şu", "o", "peki" gibi bir ifadeyle başlıyorsa,
    # ve kısaysa bu sorgudan doğrudan yeni bir ürün adı çıkarmaya çalışma, bağlama güven.
    query_starts_with_filler = any(
        processed_query_for_analysis.startswith(p) for p in ["bu ", "bunun ", "şu ", "şunun ", "o ", "onun ", "peki "]
    )
    if query_starts_with_filler and len(remaining_words_after_size_extraction) < 3 :
         entities["product"] = None # Yeni ürün adı yok, bağlama güvenilecek
         print(f"DEBUG (extract_simple_entities - Kural 2): Query='{query}', Intent='{intent}' -> Entities={entities}")
         return entities

    # Kural 3: Eğer yukarıdaki kurallar tetiklenmediyse ve niyet ürünle ilgiliyse, ürün adı çıkarmaya çalış.
    product_extraction_intents = ["fiyat_sorgulama", "ürün_bilgisi_sorma", "stok_sorgulama", "iade_sorgulama", "ürün_malzeme_sorma"]
    if intent in product_extraction_intents:
        product_related_keywords = []
        if intent == "fiyat_sorgulama":
            product_related_keywords = ["fiyatı", "fiyatını", "fiyat", "ne kadar", "ücreti"]
        elif intent in ["ürün_bilgisi_sorma", "ürün_malzeme_sorma"]:
            product_related_keywords = ["hakkında", "bilgi", "özellikleri", "detayları", "kumaşı", "içeriği", "malzeme"]
        elif intent == "stok_sorgulama":
            product_related_keywords = ["var mı", "kaldı mı", "mevcut mu", "bulunur mu", "bedeni"]
        elif intent == "iade_sorgulama":
            product_related_keywords = ["iade", "değişim"]

        min_index = len(original_query_for_product_search)
        keyword_found = False
        for kw in product_related_keywords:
            try:
                idx = original_query_for_product_search.lower().index(kw)
                if idx < min_index:
                    min_index = idx
                    keyword_found = True
            except ValueError:
                continue
        
        candidate = ""
        if keyword_found and min_index > 0:
            candidate = original_query_for_product_search[:min_index].strip()
        elif not keyword_found and len(original_query_for_product_search.split()) < 4 : 
            candidate = original_query_for_product_search.strip()

        if candidate:
            candidate = re.sub(r"[\?\.!]", "", candidate).strip()
            words_in_candidate = candidate.split()
            if not (len(words_in_candidate) == 1 and words_in_candidate[0].lower() in ["bu", "şu", "o", "peki"]):
                cleaned_candidate = " ".join(word for word in words_in_candidate if word.lower() not in ["acaba", "nedir", "için", "ile"])
                cleaned_candidate = cleaned_candidate.strip()
                if cleaned_candidate and len(cleaned_candidate) > 2 : 
                    if not cleaned_candidate.isdigit() and not re.fullmatch(r"[xsmlXSML]+", cleaned_candidate, re.IGNORECASE):
                        product_candidate = cleaned_candidate.replace("bedeni", "").strip() 
                        if not product_candidate: 
                            product_candidate = None
    
    entities["product"] = product_candidate
    print(f"DEBUG (extract_simple_entities - Kural 3 veya Sonrası): Query='{query}', Intent='{intent}' -> Entities={entities}")
    return entities

def normalize_product_name(product_name_phrase: str) -> str:
    if not product_name_phrase:
        return ""
    name = product_name_phrase.lower()
    suffixes_to_remove = ["un", "ün", "ın", "in", "u", "ü", "ı", "i", "nun", "nün", "nın", "nin"]
    cleaned_name = name
    for suffix in sorted(suffixes_to_remove, key=len, reverse=True):
        if name.endswith(suffix):
            base = name[:-len(suffix)]
            if len(base) > 2: 
                cleaned_name = base
                break 
    if cleaned_name == name and (name.endswith("u") or name.endswith("ü") or name.endswith("ı") or name.endswith("i")):
        base = name[:-1]
        if len(base) > 2:
            cleaned_name = base
            
    return cleaned_name if cleaned_name else name

def call_slm_model(query: str) -> NLUSingleAnalysis:
    global nlu_model
    if nlu_model is None:
        return NLUSingleAnalysis(slm_intent="slm_modeli_yüklenemedi", slm_entities=[], confidence_score=0.0, message="SLM (fastText) modeli yüklenemediği için analiz yapılamadı.")
    
    processed_query = query.lower().strip()
    predictions = nlu_model.predict(processed_query, k=1)
    intent_name = "tahmin_yok_slm_ile"
    confidence = 0.0
    slm_entities = [] 

    if predictions and predictions[0]: 
        if predictions[0]: 
            predicted_label_full = predictions[0][0]
            confidence = predictions[1][0]
            intent_name = predicted_label_full.replace("__label__", "")
    
    print(f"SLM MODELİ (fastText) ÇAĞRILDI: Sorgu '{query}' -> Tahmin: {intent_name}, Güven: {confidence:.4f}")
    return NLUSingleAnalysis(
        slm_intent=intent_name,
        slm_entities=slm_entities,
        confidence_score=float(f"{confidence:.4f}"),
        message="Bu yanıt fastText SLM modelinden geldi."
    )

@app.post("/process_query/", response_model=NLUResponse)
async def process_query(request: QueryRequest):
    query = request.query
    session_id = request.session_id

    if session_id is None or session_id not in active_sessions:
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {"history": [], "last_updated": datetime.now(), "last_mentioned_product": None}
        previous_query_from_session = None
        print(f"Yeni oturum başlatıldı: {session_id}")
    else:
        session_data = active_sessions[session_id]
        previous_query_from_session = session_data["history"][-1]["query"] if session_data["history"] else None
        session_data["last_updated"] = datetime.now()
        print(f"Mevcut oturum devam ediyor: {session_id}, Önceki sorgu: {previous_query_from_session}")
    
    active_sessions[session_id]["history"].append({"query": query, "timestamp": datetime.now()})
    active_sessions[session_id]["history"] = active_sessions[session_id]["history"][-5:]

    nlu_method = "regex"
    detected_intent_via_regex = None
    for intent, pattern in rules.items():
        if pattern.search(query):
            detected_intent_via_regex = intent
            break
    
    final_intent = detected_intent_via_regex
    slm_analysis_result = None

    if detected_intent_via_regex is None:
        nlu_method = "slm_fasttext"
        slm_analysis_result = call_slm_model(query)
        final_intent = slm_analysis_result.slm_intent

    current_entities = extract_simple_entities(query, final_intent)
    extracted_product_this_turn = current_entities.get("product")
    resolved_size = current_entities.get("size")

    session_data = active_sessions[session_id] 
    last_mentioned_product_from_context = session_data.get("last_mentioned_product")
    resolved_product = None 

    if extracted_product_this_turn:
        resolved_product = extracted_product_this_turn
        session_data["last_mentioned_product"] = resolved_product
        print(f"DEBUG: Ürün mevcut sorgudan alındı: '{resolved_product}'. Bağlam güncellendi. (Oturum: {session_id})")
    elif final_intent in ["stok_sorgulama", "fiyat_sorgulama", "ürün_bilgisi_sorma", "iade_sorgulama", "ürün_malzeme_sorma"] and last_mentioned_product_from_context:
        resolved_product = last_mentioned_product_from_context
        print(f"DEBUG: Mevcut sorguda ürün yok, bağlamdan ürün kullanıldı: '{resolved_product}'. (Oturum: {session_id})")
    
    # Bot yanıtı ve actionable_message için varsayılanlar
    bot_response_text = f"Üzgünüm, '{query}' isteğinizle ilgili size tam olarak yardımcı olamıyorum. Farklı bir şekilde sorabilir veya ana menüdeki seçenekleri deneyebilirsiniz."
    actionable_message = f"Anlaşılan niyet: {final_intent if final_intent else 'belirlenemedi'}."
    if resolved_product: # resolved_product None değilse ve boş string değilse
        actionable_message += f" Bahsedilen ürün: {resolved_product}."
    if resolved_size:
        actionable_message += f" Belirtilen beden: {resolved_size}."

    normalized_product_name = normalize_product_name(resolved_product if resolved_product else "")

    # Niyet bazlı yanıt üretme
    if final_intent == "kargo_bilgisi_sorma":
        shipping_info_from_db = simulated_business_data.get("business_info", {}).get("shipping_info")
        if shipping_info_from_db and shipping_info_from_db.strip():
            bot_response_text = f"Kargo politikamız: {shipping_info_from_db} Daha fazla bilgi için web sitemizi ziyaret edebilirsiniz."
        else:
            bot_response_text = (
                "Bu işletme için özel kargo bilgileri şu an mevcut değil. "
                "Genel olarak ürünlerimiz, adresimiz veya çalışma saatlerimiz hakkında bilgi alabilirsiniz. Size nasıl yardımcı olabilirim?"
            )
            if final_intent: # final_intent None değilse actionable_message'ı güncelle
                 actionable_message = f"Anlaşılan niyet: {final_intent} (ancak bu işletme için özel kargo bilgisi bulunmuyor)."

    elif final_intent == "stok_sorgulama" and normalized_product_name:
        product_data = simulated_business_data["products"].get(normalized_product_name)
        if product_data:
            link = product_data["link"]
            sizes_info = product_data.get("available_sizes_info", "Beden bilgisi ürün sayfasındadır.")
            if resolved_size:
                bot_response_text = (
                    f"'{normalized_product_name.title()}' için {resolved_size} bedeni hakkında detaylı bilgiye, "
                    f"stok durumuna ve diğer beden seçeneklerine şu adresten ulaşabilirsiniz: {link}. "
                    f"Genellikle belirtilen bedenler: {sizes_info}."
                )
                if resolved_product: # resolved_product None değilse actionable_message'ı güncelle
                    actionable_message = f"Anlaşılan niyet: {final_intent}. Bahsedilen ürün: {resolved_product}. Belirtilen beden: {resolved_size}. Yani '{resolved_product}' ürününün '{resolved_size}' bedeni stokta var mı diye soruyorsunuz."
            else:
                bot_response_text = (
                    f"'{normalized_product_name.title()}' ürünümüzün tüm beden bilgilerine ve stok durumuna "
                    f"şu adresten ulaşabilirsiniz: {link}. Genellikle belirtilen bedenler: {sizes_info}."
                )
                if resolved_product: # resolved_product None değilse actionable_message'ı güncelle
                    actionable_message = f"Anlaşılan niyet: {final_intent}. Bahsedilen ürün: {resolved_product}. Yani '{resolved_product}' ürününün stok durumunu soruyorsunuz."
        else:
            bot_response_text = (
                f"'{normalized_product_name.title()}' adında bir ürün bulamadım. "
                "Farklı bir şekilde ifade edebilir veya mevcut ürünlerimizi sorabilirsiniz."
            )
            if final_intent: # final_intent None değilse actionable_message'ı güncelle
                actionable_message = f"Anlaşılan niyet: {final_intent}. Ürün bulunamadı: {normalized_product_name}."

    elif final_intent == "fiyat_sorgulama" and normalized_product_name:
        product_data = simulated_business_data["products"].get(normalized_product_name)
        if product_data:
            link = product_data["link"]
            price = product_data.get("price", "Fiyat bilgisi için lütfen ürün sayfasını ziyaret edin.")
            bot_response_text = (
                f"'{normalized_product_name.title()}' ürünümüzün fiyatı: {price}. "
                f"Daha fazla bilgi ve satın almak için: {link}"
            )
        else:
            bot_response_text = f"'{normalized_product_name.title()}' adında bir ürün bulamadım."
            if final_intent: # final_intent None değilse actionable_message'ı güncelle
                actionable_message = f"Anlaşılan niyet: {final_intent}. Ürün bulunamadı: {normalized_product_name}."

    elif final_intent == "ürün_bilgisi_sorma" and normalized_product_name:
        product_data = simulated_business_data["products"].get(normalized_product_name)
        if product_data:
            link = product_data["link"]
            description = product_data.get("description", "Açıklama mevcut değil.")
            material = product_data.get("material_composition", "")
            response_parts = [f"'{normalized_product_name.title()}': {description}"]
            if material:
                response_parts.append(f"Malzeme içeriği: {material}.")
            response_parts.append(f"Daha fazla detay için: {link}")
            bot_response_text = " ".join(response_parts)
        else:
            bot_response_text = f"'{normalized_product_name.title()}' adında bir ürün bulamadım."
            if final_intent: # final_intent None değilse actionable_message'ı güncelle
                actionable_message = f"Anlaşılan niyet: {final_intent}. Ürün bulunamadı: {normalized_product_name}."
            
    elif final_intent == "ürün_malzeme_sorma" and normalized_product_name: 
        product_data = simulated_business_data["products"].get(normalized_product_name)
        if product_data:
            link = product_data["link"]
            material = product_data.get("material_composition", "Bu ürün için özel malzeme bilgisi bulunmamaktadır.")
            bot_response_text = (
                f"'{normalized_product_name.title()}' ürünümüzün malzeme içeriği: {material}. "
                f"Daha fazla bilgi için ürün sayfasını ziyaret edebilirsiniz: {link}"
            )
        else:
            bot_response_text = f"'{normalized_product_name.title()}' adında bir ürün bulamadım."
            if final_intent: # final_intent None değilse actionable_message'ı güncelle
                actionable_message = f"Anlaşılan niyet: {final_intent}. Ürün bulunamadı: {normalized_product_name}."

    elif final_intent == "lokasyon_sorma":
        address = simulated_business_data.get("business_info", {}).get("store_address", "Adres bilgimiz şu an mevcut değil.")
        maps_link = simulated_business_data.get("business_info", {}).get("maps_link")
        response_parts = [f"Adresimiz: {address}."]
        if maps_link:
            response_parts.append(f"Harita üzerinden ulaşmak için: {maps_link}")
        bot_response_text = " ".join(response_parts)
        
    elif final_intent == "tel_no_sorma":
        phone = simulated_business_data.get("business_info", {}).get("phone", "Telefon numaramız kayıtlı değil.")
        whatsapp = simulated_business_data.get("business_info", {}).get("whatsapp_number")
        email = simulated_business_data.get("business_info", {}).get("email")
        response_parts = [f"Bize ulaşabileceğiniz telefon numaramız: {phone}."]
        if whatsapp:
            response_parts.append(f"WhatsApp hattımız: {whatsapp}.")
        if email:
            response_parts.append(f"E-posta adresimiz: {email}.")
        bot_response_text = " ".join(response_parts)

    elif final_intent == "iade_sorgulama":
        return_policy = simulated_business_data.get("business_info", {}).get("return_policy", "İade politikamız hakkında detaylı bilgi için lütfen web sitemizi ziyaret edin veya bizimle iletişime geçin.")
        bot_response_text = f"İade koşullarımız: {return_policy}"

    elif final_intent == "calisma_saatleri_sorma":
        opening_hours = simulated_business_data.get("business_info", {}).get("opening_hours", "Çalışma saatlerimiz hakkında bilgi bulunmamaktadır.")
        bot_response_text = f"Çalışma saatlerimiz: {opening_hours}."

    elif final_intent == "odeme_yontemleri_sorma":
        payment_options = simulated_business_data.get("business_info", {}).get("payment_options", "Ödeme yöntemlerimiz hakkında bilgi bulunmamaktadır.")
        bot_response_text = f"Kabul ettiğimiz ödeme yöntemleri: {payment_options}."
    
    elif final_intent == "selamlama":
        bot_response_text = "Merhaba! Size nasıl yardımcı olabilirim?"
        if final_intent: # final_intent None değilse actionable_message'ı güncelle
            actionable_message = "Anlaşılan niyet: selamlama." 
        
    elif final_intent == "tesekkur":
        bot_response_text = "Rica ederim! Başka bir konuda yardımcı olabilir miyim?"
        if final_intent: # final_intent None değilse actionable_message'ı güncelle
            actionable_message = "Anlaşılan niyet: tesekkur."

    # Yanıtı Oluştur
    if nlu_method == "slm_fasttext":
        return NLUResponse(
            original_query=query, session_id=session_id, nlu_method=nlu_method,
            analysis=slm_analysis_result, previous_query_in_session=previous_query_from_session,
            resolved_product=resolved_product, resolved_size=resolved_size, actionable_message=actionable_message,
            bot_response=bot_response_text
        )
    else: # nlu_method == "regex"
        return NLUResponse(
            original_query=query, session_id=session_id, nlu_method=nlu_method,
            detected_intent=final_intent, previous_query_in_session=previous_query_from_session,
            resolved_product=resolved_product, resolved_size=resolved_size, actionable_message=actionable_message,
            bot_response=bot_response_text
        )

@app.get("/")
async def read_root():
    return {"message": "NLU API'sine hoş geldiniz! Oturum, bağlam ve dinamik yanıt üretme eklendi."}
