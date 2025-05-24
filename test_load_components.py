from pathlib import Path
import fasttext
import zeyrek
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH_STR = "nlu_model.bin" # main.py'deki gibi
MODEL_PATH = Path(MODEL_PATH_STR).resolve()

print(f"--- Test Yükleme Scripti Başladı ---")
print(f"NLU Model Yolu (resolve edilmiş): {MODEL_PATH}")

# Zeyrek yükleme testi
try:
    print(f"Zeyrek yükleniyor...")
    morphology = zeyrek.MorphAnalyzer()
    print(f"Zeyrek başarıyla yüklendi: {morphology}")
    logger.info("Zeyrek başarıyla yüklendi.")
except Exception as e:
    print(f"ZEYREK YÜKLEME HATASI: {e}")
    logger.error(f"Zeyrek yükleme hatası: {e}", exc_info=True)

# FastText model yükleme testi
if MODEL_PATH.exists():
    if MODEL_PATH.is_file():
        print(f"FastText modeli '{MODEL_PATH}' yükleniyor...")
        try:
            nlu_model = fasttext.load_model(str(MODEL_PATH))
            print(f"FastText modeli başarıyla yüklendi: {nlu_model}")
            logger.info(f"FastText modeli '{MODEL_PATH}' başarıyla yüklendi.")
        except Exception as e:
            print(f"FASTTEXT MODEL YÜKLEME HATASI ({MODEL_PATH}): {e}")
            logger.error(f"FastText model yükleme hatası ({MODEL_PATH}): {e}", exc_info=True)
    else:
        print(f"HATA: Model yolu '{MODEL_PATH}' bir dosya değil.")
        logger.error(f"HATA: Model yolu '{MODEL_PATH}' bir dosya değil.")
else:
    print(f"HATA: Model dosyası '{MODEL_PATH}' bulunamadı.")
    logger.error(f"HATA: Model dosyası '{MODEL_PATH}' bulunamadı.")

print(f"--- Test Yükleme Scripti Bitti ---")