#!/usr/bin/env python3
"""
Ürün arama fonksiyonlarını test eden basit script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import search_products, get_product_categories, format_product_list, format_product_info, get_product_by_code

def test_product_functions():
    print("🧪 Ürün Fonksiyonları Test Ediliyor...")
    print("=" * 50)
    
    # 1. Kategorileri test et
    print("\n1️⃣ Kategoriler:")
    categories = get_product_categories()
    for i, cat in enumerate(categories, 1):
        print(f"   {i}. {cat}")
    
    # 2. Genel arama test et
    print("\n2️⃣ Genel Arama (ilk 3 ürün):")
    products = search_products("", limit=3)
    print(format_product_list(products))
    
    # 3. Spesifik arama test et
    print("\n3️⃣ 'Pijama' Araması:")
    products = search_products("pijama", limit=3)
    print(format_product_list(products))
    
    # 4. Renk araması test et
    print("\n4️⃣ 'Siyah' Araması:")
    products = search_products("siyah", limit=3)
    print(format_product_list(products))
    
    # 5. Ürün kodu ile arama
    print("\n5️⃣ Ürün Kodu ile Arama (18K18154):")
    product = get_product_by_code("18K18154")
    if product:
        print(format_product_info(product))
    else:
        print("Ürün bulunamadı")
    
    # 6. Fiyat filtresi test et
    print("\n6️⃣ 2000₺ Altı Ürünler:")
    products = search_products("", max_price=2000, limit=3)
    print(format_product_list(products))
    
    # 7. Kategori filtresi test et
    print("\n7️⃣ Gecelikler Kategorisi:")
    products = search_products("", category="Gecelikler", limit=3)
    print(format_product_list(products))

if __name__ == "__main__":
    test_product_functions() 