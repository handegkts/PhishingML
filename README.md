# Phishing Tespiti için Makine Öğrenmesi Uygulaması

Bu proje, e-posta ve URL'lerdeki phishing (oltalama) saldırılarını tespit etmek için geliştirilmiş bir makine öğrenmesi uygulamasıdır. Farklı sınıflandırma algoritmaları kullanarak şüpheli içerikleri analiz eder ve kullanıcı dostu bir arayüz sunar.

## Özellikler

- **Çoklu Model Desteği**:
  - Random Forest
  - Destek Vektör Makineleri (SVM)
  - Karar Ağaçları
  - Naive Bayes
  - K-En Yakın Komşu (KNN)

## Kurulum ve Çalıştırma

1. Gereksinimleri yükleyin:
   ```bash
   python -m pip install -r requirements.txt
   ```

2. Uygulamayı başlatın:
   ```bash
   python app.py
   ```

3. Tarayıcınızda `http://localhost:5051` adresine gidin.

## Kullanım

1. **Model Eğitimi**:
   - Ana sayfadan "Model Eğitimi" bölümüne gidin
   - Eğitmek istediğiniz modeli seçin
   - Eğitim parametrelerini ayarlayın
   - "Eğitime Başla" butonuna tıklayın

2. **Kayıt Edilmiş Eğitilmiş Model Eğitimleri**:
   - Ana sayfadan "Kayıt Edilmiş Eğitilmiş Model Eğitimleri" bölümüne gidin
   - Kayıt edilmiş eğitilmiş model eğitimleri ile ilgili detayları görüntüleyin
   - Kayıt edilmiş model eğitimler phishingml.db veritabanında bulunur.
   - instances/models dizini altında model eğitimlerine ait dosyalar bulunmaktadır.
   - Eğitim aşamasında yüklenen dosyalar instances/upload dizini altında bulunmaktadır.


# Veri Setleri

Projede kullanılan örnek veri setleri `instances/sample_datasets` dizini altında bulunmaktadır. Veri setleri şunlardır:

1. **Email Phishing Dataset**
   - Kaynak: [Kaggle](https://www.kaggle.com/datasets/ethancratchley/email-phishing-dataset)
   - Kayıt Sayısı: 524,846
   - Dosya: instances/sample_datasets/email_phishing_dataset.csv

2. **Web Page Phishing Dataset**
   - Kaynak: [Kaggle](https://www.kaggle.com/datasets/danielfernandon/web-page-phishing-dataset)
   - Kayıt Sayısı: 100,077
   - Dosya: instances/sample_datasets/web_page_phishing_dataset.csv

3. **Phishing Websites**
   - Kaynak: [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites)
   - Kayıt Sayısı: 11,055
   - Dosya: instances/sample_datasets/phishing_websites.csv