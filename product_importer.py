import sqlite3
import csv
import re

def clean_price(price_str):
    """Fiyat string'ini temizle ve float'a çevir"""
    if not price_str or price_str.strip() == '':
        return None
    
    # Virgülü nokta ile değiştir ve sayısal olmayan karakterleri temizle
    cleaned = price_str.replace(',', '.').replace('"', '').strip()
    
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None

def import_products_from_csv():
    """CSV dosyasından ürünleri veritabanına aktar"""
    
    # Veritabanı bağlantısı
    conn = sqlite3.connect('chatbot_data.db')
    cursor = conn.cursor()
    
    # Ürünler tablosunu oluştur
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_code TEXT UNIQUE NOT NULL,
            product_name TEXT NOT NULL,
            color TEXT,
            original_price REAL,
            discount_rate REAL,
            discounted_price REAL,
            category TEXT DEFAULT 'Kadın Giyim',
            description TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # CSV dosyasını oku ve veritabanına aktar
    imported_count = 0
    skipped_count = 0
    
    with open('Ürün Bilgisi Raporu.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            product_code = row['Ürün Kodu'].strip()
            product_name = row['Ürün Adı'].strip()
            color = row['Renk'].strip()
            
            # Fiyatları temizle
            original_price = clean_price(row['Monamise Satış Fiyatı'])
            discount_rate = clean_price(row['İndirim Oranı (%)'])
            discounted_price = clean_price(row['Monamise İndirimli Net Satış Fiyatı'])
            
            # Fiyatı olmayan ürünleri şimdilik atla (deneme aşaması)
            if original_price is None:
                skipped_count += 1
                continue
            
            # Kategori belirleme (ürün adından)
            category = determine_category(product_name)
            
            # Açıklama oluştur
            description = create_description(product_name, color, original_price, discounted_price, discount_rate)
            
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO products 
                    (product_code, product_name, color, original_price, discount_rate, 
                     discounted_price, category, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (product_code, product_name, color, original_price, discount_rate,
                      discounted_price, category, description))
                
                imported_count += 1
                
            except Exception as e:
                print(f"Hata: {product_code} - {e}")
                skipped_count += 1
    
    conn.commit()
    conn.close()
    
    print(f"✅ İçe aktarma tamamlandı!")
    print(f"📦 İçe aktarılan ürün sayısı: {imported_count}")
    print(f"⏭️ Atlanan ürün sayısı: {skipped_count}")
    
    return imported_count, skipped_count

def determine_category(product_name):
    """Ürün adından kategori belirle"""
    name_lower = product_name.lower()
    
    if any(word in name_lower for word in ['pijama', 'takım', 'alt üst']):
        return 'Pijama Takımları'
    elif any(word in name_lower for word in ['gecelik', 'gece']):
        return 'Gecelikler'
    elif any(word in name_lower for word in ['sabahlık', 'sabah']):
        return 'Sabahlıklar'
    elif any(word in name_lower for word in ['şort', 'short']):
        return 'Şort Takımları'
    elif any(word in name_lower for word in ['hamile', 'lohusa']):
        return 'Hamile & Lohusa'
    elif any(word in name_lower for word in ['elbise']):
        return 'Elbiseler'
    elif any(word in name_lower for word in ['tulum']):
        return 'Tulumlar'
    elif any(word in name_lower for word in ['kapri']):
        return 'Kapri Takımları'
    else:
        return 'Kadın Giyim'

def create_description(name, color, original_price, discounted_price, discount_rate):
    """Ürün açıklaması oluştur"""
    desc_parts = []
    
    # Renk bilgisi
    if color:
        desc_parts.append(f"Renk: {color}")
    
    # Fiyat bilgisi
    if original_price and discounted_price:
        desc_parts.append(f"Fiyat: {original_price:.2f}₺ yerine {discounted_price:.2f}₺")
        if discount_rate:
            desc_parts.append(f"%{discount_rate:.0f} indirim")
    elif original_price:
        desc_parts.append(f"Fiyat: {original_price:.2f}₺")
    
    # Özellikler (ürün adından çıkar)
    features = extract_features(name)
    if features:
        desc_parts.extend(features)
    
    return " | ".join(desc_parts)

def extract_features(product_name):
    """Ürün adından özellikler çıkar"""
    features = []
    name_lower = product_name.lower()
    
    # Özellik anahtar kelimeleri
    feature_keywords = {
        'dantelli': 'Dantel detaylı',
        'dekolteli': 'Dekolte',
        'düğmeli': 'Düğmeli',
        'askılı': 'Askılı',
        'kısa kollu': 'Kısa kollu',
        'uzun kollu': 'Uzun kollu',
        'v yaka': 'V yaka',
        'büyük beden': 'Büyük beden',
        'hamile': 'Hamile uygun',
        'lohusa': 'Lohusa uygun',
        'saten': 'Saten kumaş',
        'kadife': 'Kadife kumaş',
        'brode': 'Brode işlemeli'
    }
    
    for keyword, feature in feature_keywords.items():
        if keyword in name_lower:
            features.append(feature)
    
    return features[:3]  # En fazla 3 özellik

def test_database():
    """Veritabanını test et"""
    conn = sqlite3.connect('chatbot_data.db')
    cursor = conn.cursor()
    
    # Toplam ürün sayısı
    cursor.execute('SELECT COUNT(*) FROM products')
    total_count = cursor.fetchone()[0]
    
    # Kategorilere göre dağılım
    cursor.execute('SELECT category, COUNT(*) FROM products GROUP BY category ORDER BY COUNT(*) DESC')
    categories = cursor.fetchall()
    
    # Fiyat aralıkları
    cursor.execute('SELECT MIN(original_price), MAX(original_price), AVG(original_price) FROM products WHERE original_price IS NOT NULL')
    price_stats = cursor.fetchone()
    
    print(f"\n📊 VERİTABANI İSTATİSTİKLERİ")
    print(f"{'='*50}")
    print(f"Toplam ürün sayısı: {total_count}")
    print(f"\nKategoriler:")
    for category, count in categories:
        print(f"  • {category}: {count} ürün")
    
    if price_stats[0]:
        print(f"\nFiyat İstatistikleri:")
        print(f"  • En düşük fiyat: {price_stats[0]:.2f}₺")
        print(f"  • En yüksek fiyat: {price_stats[1]:.2f}₺")
        print(f"  • Ortalama fiyat: {price_stats[2]:.2f}₺")
    
    # Örnek ürünler
    cursor.execute('SELECT product_code, product_name, color, original_price, discounted_price FROM products LIMIT 5')
    sample_products = cursor.fetchall()
    
    print(f"\nÖrnek Ürünler:")
    for product in sample_products:
        code, name, color, orig_price, disc_price = product
        if orig_price and disc_price:
            print(f"  • {code}: {name[:40]}... ({color}) - {orig_price:.0f}₺ → {disc_price:.0f}₺")
        else:
            print(f"  • {code}: {name[:40]}... ({color})")
    
    conn.close()

if __name__ == "__main__":
    print("🚀 Ürün içe aktarma işlemi başlıyor...")
    imported, skipped = import_products_from_csv()
    test_database()
    print(f"\n✨ İşlem tamamlandı! {imported} ürün başarıyla aktarıldı.") 