import fasttext
import os # Dosya yollarını birleştirmek için

# Proje ana dizinini alalım (bu betiğin bulunduğu dizin)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modelin eğitileceği veri dosyası ve kaydedileceği dosya adı
# Dosya yollarını BASE_DIR ile birleştirerek tam yol elde ediyoruz
training_data_path = os.path.join(BASE_DIR, "train_nlu.txt")
model_save_path = os.path.join(BASE_DIR, "nlu_model.bin") # .ftz uzantısı sıkıştırılmış model için kullanılabilir

print(f"Eğitim verisi '{training_data_path}' kullanılıyor...")
print(f"Model '{model_save_path}' olarak kaydedilecek...")

try:
    # Modeli eğit
    # 'supervised' metodu, etiketlenmiş verilerle sınıflandırma modeli eğitir.
    # wordNgrams=2: Kelime n-gramlarını (bu durumda bi-gramları) da özellik olarak kullanır, bu genellikle daha iyi sonuç verir.
    # epoch=25: Eğitim verisi üzerinden 25 kez geçer. Daha fazla veriyle veya daha iyi sonuç için artırılabilir.
    # lr=0.5: Öğrenme oranı (fastText varsayılanı 0.1, supervised için 0.5-1.0 arası denenebilir).
    # dim=100: Kelime vektörlerinin boyutu. Veri miktarına ve karmaşıklığa göre ayarlanabilir (örn: 50-300).
    # loss='softmax': Çoklu sınıflandırma için genellikle 'softmax' veya 'hs' (hierarchical softmax) kullanılır.
    # thread=4: Eğitim için kullanılacak thread sayısı. Makinenizin çekirdek sayısına göre ayarlayabilirsiniz.
    model = fasttext.train_supervised(
        input=training_data_path,
        wordNgrams=2,
        epoch=25,
        lr=0.5,
        dim=100,
        loss='softmax',
        thread=4,
        verbose=2 # Eğitim sürecinde daha fazla bilgi gösterir (0-2)
    )

    # Eğitilen modeli kaydet
    model.save_model(model_save_path)

    print(f"Model başarıyla eğitildi ve '{model_save_path}' olarak kaydedildi.")

    # İsteğe bağlı: Modelin performansını değerlendirmek için bir test veri seti kullanılabilir.
    # Eğer bir test_nlu.txt dosyanız varsa (train_nlu.txt ile aynı formatta ama farklı örneklerle):
    # test_data_path = os.path.join(BASE_DIR, "test_nlu.txt")
    # if os.path.exists(test_data_path):
    #     print("Test verisi üzerinde değerlendirme yapılıyor...")
    #     # N: Örnek sayısı, P@1: Precision at 1, R@1: Recall at 1
    #     result = model.test(test_data_path)
    #     precision = result[1]
    #     recall = result[2]
    #     f1_score = 0
    #     if (precision + recall) > 0:
    #         f1_score = 2 * (precision * recall) / (precision + recall)
        
    #     print(f"Test Örnek Sayısı: {result[0]}")
    #     print(f"Precision @1: {precision:.4f}")
    #     print(f"Recall @1: {recall:.4f}")
    #     print(f"F1 Score @1: {f1_score:.4f}")
    # else:
    #     print("Test verisi bulunamadı, değerlendirme atlandı.")

except Exception as e:
    print(f"Model eğitimi sırasında bir hata oluştu: {e}")
    print("Lütfen 'train_nlu.txt' dosyasının varlığını ve içeriğini kontrol edin.")