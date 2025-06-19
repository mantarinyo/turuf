import requests
from bs4 import BeautifulSoup
import re

def fetch_and_format_wiktionary_frequency_list(url, output_filename="turkish_frequency_dictionary.txt"):
    """
    Verilen Wiktionary URL'sinden Türkçe kelime frekans listesini çeker
    ve 'kelime<boşluk>frekans' formatında bir dosyaya kaydeder.
    """
    try:
        response = requests.get(url)
        response.raise_for_status() # HTTP hatalarını kontrol et
        print(f"'{url}' adresinden sayfa başarıyla çekildi.")
    except requests.exceptions.RequestException as e:
        print(f"HATA: Sayfa çekilemedi. {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')

    # Genellikle bu tür listeler tablolarda bulunur.
    # Sayfadaki ana içerik tablosunu bulmaya çalışalım.
    # Wiktionary sayfasındaki tablo yapısı genellikle bir 'table' etiketi içindedir.
    # Sınıfı 'wikitable sortable' olan tabloyu arayabiliriz veya daha genel bir arama yapabiliriz.
    data_table = soup.find('table', {'class': 'wikitable'}) # Veya sadece 'table'

    if not data_table:
        data_table = soup.find('table') # İlk bulduğu tabloyu dene
        if not data_table:
            print("HATA: Sayfada frekans listesi içeren bir tablo bulunamadı.")
            return

    print("Tablo bulundu, veriler işleniyor...")
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            rows = data_table.find_all('tr')
            header_skipped = False
            count_written = 0

            for row in rows:
                cols = row.find_all('td')
                if not cols: # Başlık satırı veya boş satır olabilir
                    if not header_skipped and row.find_all('th'): # Başlık satırını atla
                        header_skipped = True
                        print("Başlık satırı atlandı.")
                    continue

                # Wiktionary formatı genellikle: Rank, Word, Count, (bazen Percent)
                # Kelime genellikle 2. sütunda (index 1), Frekans 3. sütunda (index 2) olur.
                # Sayfanın yapısına göre bu indexler değişebilir.
                # Verdiğin linkteki yapıya göre:
                # cols[0] = Rank, cols[1] = Word, cols[2] = Count
                if len(cols) >= 3:
                    word = cols[1].get_text(strip=True)
                    count_str = cols[2].get_text(strip=True)
                    
                    # Frekans sayısındaki virgülleri ve olası diğer karakterleri temizle
                    count_str_cleaned = re.sub(r'[,\.]', '', count_str) # Virgül ve noktaları kaldır
                    
                    if word and count_str_cleaned.isdigit():
                        f.write(f"{word.lower()}\t{count_str_cleaned}\n") # Kelimeyi küçük harf yap ve TAB ile ayır
                        count_written += 1
                    else:
                        print(f"UYARI: Satır atlandı (geçersiz kelime veya frekans): Kelime='{word}', Frekans='{count_str}'")
                else:
                    print(f"UYARI: Satırda yeterli sütun yok, atlanıyor: {row.get_text(strip=True)[:50]}...")
            
            print(f"İşlem tamamlandı. '{output_filename}' dosyasına {count_written} kelime yazıldı.")

    except Exception as e:
        print(f"HATA: Dosya yazma veya veri işleme sırasında bir sorun oluştu: {e}")

if __name__ == "__main__":
    wiktionary_url = "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Turkish_WordList_10K"
    fetch_and_format_wiktionary_frequency_list(wiktionary_url)
    
    # İstersen ilk 1000 kelimeyi almak için şöyle bir şey yapabilirsin:
    # fetch_and_format_wiktionary_frequency_list(wiktionary_url, output_filename="turkish_freq_1000.txt")
    # Sonrasında oluşan dosyadan ilk 1000 satırı alıp asıl dosyanı oluşturabilirsin,
    # ya da betiği ilk N kelimeyi alacak şekilde de güncelleyebiliriz.
    # Şimdilik tüm listeyi çekiyor.
