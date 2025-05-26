# PhishingML - Makine Öğrenmesi ile Phishing Tespiti

Bu proje, çeşitli makine öğrenmesi algoritmalarını kullanarak phishing web sitelerini tespit eden bir web uygulamasıdır. Flask framework'ü üzerine inşa edilmiş olup, CSV formatında veri setleri yükleyerek modelleri eğitmeye ve test etmeye olanak tanır.

## Özellikler

- Çeşitli makine öğrenmesi modellerini eğitme:
  - XGBoost
  - SVM (Destek Vektör Makineleri)
  - Decision Tree
  - Naive Bayes
  - KNN (K-En Yakın Komşu)
- CSV formatında veri setlerini yükleme ve işleme
- Eğitilmiş modelleri kullanarak phishing testi yapma
- Kullanıcı dostu web arayüzü

## Kurulum

1. Proje deposunu klonlayın:
```
git clone https://github.com/yourusername/PhishingML.git
cd PhishingML
```

2. Sanal ortam oluşturun ve etkinleştirin:
```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Gerekli paketleri yükleyin:
```
pip install -r requirements.txt
```

4. Uygulamayı çalıştırın:
```
python app.py
```

5. Tarayıcınızda `http://127.0.0.1:5000` adresine gidin

## Kullanım

### Model Eğitimi

1. "Model Eğitimi" sayfasına gidin
2. CSV formatında bir phishing veri seti yükleyin
3. Eğitmek istediğiniz model türünü seçin
4. "Modeli Eğit" düğmesine tıklayın
5. Eğitim tamamlandığında doğruluk oranı görüntülenecektir

### Phishing Testi

1. "Phishing Testi" sayfasına gidin
2. Test etmek istediğiniz modeli seçin
3. İlgili özellikleri girin
4. "Testi Çalıştır" düğmesine tıklayın
5. Sonuç sayfanın üst kısmında görüntülenecektir

## Veri Seti Gereksinimleri

- CSV formatında olmalıdır
- Veri setinde özellik değerleri sayısal olmalıdır
- Son sütun hedef değişkeni içermelidir (1: Phishing, 0: Güvenli)
- Sütun başlıkları olmalıdır (özellik isimleri)
- Eksik değer içermemesi önerilir

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın. 