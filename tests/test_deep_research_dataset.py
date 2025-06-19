import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

# Kapsamlı veri kümesi (örnek olarak ilk 2 kayıt, devamı eklenmeli)
dataset = [
    {
        "kategori": "stok",
        "ana_soru": "Bu elbisenin M bedeni var mı?",
        "varyasyonlar": [
            "M stok kaldımı",
            "m beden var mi",
            "Medium var mı ya"
        ],
        "ideal_cevap": "Merhaba! Evet, M bedenimiz şu an stokta mevcut. Sipariş vermek isterseniz size memnuniyetle yardımcı olabiliriz.",
        "notlar": ""
    },
    {
        "kategori": "stok",
        "ana_soru": "Bu montun siyahı tükendi mi, tekrar gelecek mi?",
        "varyasyonlar": [
            "Montun siyahı gelirmi yine?",
            "bunun siyahı gelecek mi?",
            "siyahı gelir mi ya tekrar"
        ],
        "ideal_cevap": "Merhabalar, siyah renk şu anda maalesef stokta yok. Önümüzdeki hafta tekrar stokta olmasını bekliyoruz. Geldiğinde haber vermemizi isterseniz sizi listemize ekleyebiliriz.",
        "notlar": ""
    },
    # ... (Tüm veri kümesi buraya eklenmeli) ...
]

@pytest.mark.parametrize("soru, beklenen_cevap", [
    (item["ana_soru"], item["ideal_cevap"]) for item in dataset
])
def test_ideal_cevap_ana_sorular(client, soru, beklenen_cevap):
    response = client.post("/chat", json={"question": soru})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == beklenen_cevap

@pytest.mark.parametrize("varyasyon, beklenen_cevap", [
    (varyasyon, item["ideal_cevap"]) 
    for item in dataset 
    for varyasyon in item["varyasyonlar"]
])
def test_ideal_cevap_varyasyonlar(client, varyasyon, beklenen_cevap):
    response = client.post("/chat", json={"question": varyasyon})
    assert response.status_code == 200
    result = response.json()
    assert "answer" in result
    assert result["answer"] == beklenen_cevap 