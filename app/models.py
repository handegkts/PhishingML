import os
import joblib
import pandas as pd
import numpy as np
import json  # JSON işlemleri için açıkça import ediyoruz
from datetime import datetime, timedelta
import pytz  # Zaman dilimi işlemleri için
import hashlib  # Veri seti hash hesaplaması için eklendi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix # Detaylı loglama için eklendi

# Modeller
# XGBoost kaldırıldı, çünkü yükleme sorunları yaşıyoruz
# from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from flask import current_app
from app import db

# Türkiye zaman dilimini tanımla
turkey_timezone = pytz.timezone('Europe/Istanbul')

def get_current_turkey_time():
    """Mevcut zamanı Türkiye saati (UTC+3) olarak döndürür."""
    return datetime.now(turkey_timezone)

class TrainedModel(db.Model):
    __tablename__ = 'trained_models'
    
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    model_name_display = db.Column(db.String(100), nullable=False, unique=True)
    dataset_name = db.Column(db.String(200), nullable=False)
    dataset_hash = db.Column(db.String(64), nullable=True)  # Veri seti hash değeri için yeni alan
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    training_time = db.Column(db.Float, nullable=False)
    trained_at = db.Column(db.DateTime, default=get_current_turkey_time)
    model_path = db.Column(db.String(500), nullable=False)
    random_seed = db.Column(db.Integer, nullable=True)  # Random seed değeri için yeni alan
    test_data_path = db.Column(db.String(500), nullable=True)
    training_data_path = db.Column(db.String(500), nullable=True)
    icon_class = db.Column(db.String(100), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    
    # Eğitim ve test veri sayıları
    train_samples_count = db.Column(db.Integer, nullable=True)
    test_samples_count = db.Column(db.Integer, nullable=True)
    training_data_json = db.Column(db.Text, nullable=True)  # Eğitim verisi JSON formatında
    test_data_json = db.Column(db.Text, nullable=True)  # Test verisi JSON formatında
    training_logs_json = db.Column(db.Text, nullable=True) # Eğitim logları JSON formatında
    model_params_json = db.Column(db.Text, nullable=True) # Model parametreleri JSON formatında
    
    def __repr__(self):
        return f"<TrainedModel {self.model_name} ({self.trained_at}), Train: {self.train_samples_count}, Test: {self.test_samples_count}>"

def get_model_icon_class(model_type):
    icon_classes = {
        'random_forest': 'bi bi-diagram-3 text-success',
        'svm': 'bi bi-grid-3x3-gap text-info',
        'decision_tree': 'bi bi-diagram-2 text-warning',
        'naive_bayes': 'bi bi-calculator text-danger',
        'knn': 'bi bi-people text-primary',
        'xgboost': 'bi bi-lightning text-primary'
    }
    return icon_classes.get(model_type, 'bi bi-cpu text-secondary')

def train_model(file_path, model_type='random_forest', model_params=None):
    """
    Belirtilen CSV dosyasını kullanarak seçilen modeli eğitir.
    
    Args:
        file_path: Eğitim verilerini içeren CSV dosyasının yolu
        model_type: Eğitilecek model tipi ('random_forest', 'svm', 'decision_tree', 'naive_bayes', 'knn')
        model_params: Model için hiperparametreler (sözlük)
    
    Returns:
        dict: Eğitim metrikleri, model bilgileri ve logları içeren sözlük
        str: Kaydedilen modelin yolu
    """
    logs = []
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model eğitim süreci başlatıldı.")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Veri seti yükleniyor: {os.path.basename(file_path)}")
    # Veriyi yükle
    df = pd.read_csv(file_path)
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Veri seti başarıyla yüklendi. Satır: {len(df)}, Sütun: {len(df.columns)}")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DEBUG: Veri setinin ilk 5 satırı:\\n{df.head().to_string()}")
    
    # Veri seti hash değerini hesapla
    try:
        # Veri setinin ilk 1000 satırını (veya tamamını) kullanarak hash oluştur
        sample_for_hash = df.head(min(1000, len(df)))
        dataset_str = sample_for_hash.to_string()
        dataset_hash = hashlib.sha256(dataset_str.encode('utf-8')).hexdigest()
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Veri seti hash değeri hesaplandı: {dataset_hash}")
    except Exception as e:
        dataset_hash = None
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Veri seti hash değeri hesaplanırken hata oluştu: {str(e)}")
    
    # Eksik değerleri işle
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: {dropped_rows} satır eksik değerler nedeniyle kaldırıldı.")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Veri ön işleme tamamlandı. Kullanılabilir satır: {len(df)}")
    
    # Özellikler ve hedef değişkeni ayır
    # Hedef değişkenin son sütunda olduğunu varsayalım
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Özellikler (X) ve hedef (y) değişkenleri ayrıldı.")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DEBUG: Özellik sütunları (X.columns): {X.columns.tolist()}")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DEBUG: X şekli: {X.shape}, y şekli: {y.shape}")
    
    # Özellik adlarını kaydet (tahmin için)
    feature_names = X.columns.tolist()
    
    # Veri tiplerini kontrol et ve metin sütunlarını temizle
    non_numeric_columns = []
    for col in X.columns:
        if X[col].dtype == 'object':  # Eğer sütun metin türündeyse
            non_numeric_columns.append(col)
            logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Metin türünde sütun tespit edildi: '{col}'. Bu sütun kaldırılacak.")
    
    if non_numeric_columns:
        # Metin sütunlarını kaldır
        X = X.drop(columns=non_numeric_columns)
        # Özellik adlarını güncelle
        feature_names = X.columns.tolist()
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: {len(non_numeric_columns)} metin sütunu kaldırıldı. Yeni özellik sayısı: {len(feature_names)}")
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DEBUG: Yeni özellik sütunları: {feature_names}")
    
    # Tüm sütunları sayısala çevir
    try:
        X = X.apply(pd.to_numeric, errors='coerce')
        # NaN değerleri 0 ile doldur
        X = X.fillna(0)
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Tüm sütunlar sayısal değerlere dönüştürüldü.")
    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Sütunlar sayısala dönüştürülürken hata: {str(e)}")
    
    # Veriyi eğitim ve test setlerine böl
    # model_params'dan train_test_split oranını al, yoksa varsayılan kullan
    # train_test_split parametresi eğitim verisi yüzdesini belirtir
    train_size_param = float(model_params.get('train_test_split', 80)) / 100 if model_params else 0.8
    test_size_param = 1.0 - train_size_param  # Test oranı, eğitim oranının tümleyeni
    
    # Random seed değerini al, yoksa varsayılan 42 kullan
    random_seed = int(model_params.get('random_seed', 42)) if model_params else 42
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Veri eğitim ve test setlerine bölünüyor. Eğitim oranı: {train_size_param*100:.0f}%, Test oranı: {test_size_param*100:.0f}%, Random Seed: {random_seed}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_param, random_state=random_seed)
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DEBUG: X_train şekli: {X_train.shape}, y_train şekli: {y_train.shape}")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DEBUG: X_test şekli: {X_test.shape}, y_test şekli: {y_test.shape}")
    
    # Eğitim ve test veri sayılarını sakla
    train_samples_count = len(X_train)
    test_samples_count = len(X_test)
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Eğitim veri sayısı: {train_samples_count}, Test veri sayısı: {test_samples_count}")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DEBUG: Değerler doğru mu? train_samples_count={train_samples_count} (X_train) ve test_samples_count={test_samples_count} (X_test)")
    
    # Eğitim ve test verilerinden örnekleri JSON formatında sakla (en fazla 10 örnek)
    sample_train_df = pd.DataFrame(X_train.iloc[:10].copy())
    sample_train_df['target'] = y_train.iloc[:10].values if hasattr(y_train, 'iloc') else y_train[:10]
    try:
        train_data_json = sample_train_df.to_json(orient='records')
    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Eğitim verisi JSON dönüşümünde hata: {e}")
        train_data_json = json.dumps([])
    
    sample_test_df = pd.DataFrame(X_test.iloc[:10].copy())
    sample_test_df['target'] = y_test.iloc[:10].values if hasattr(y_test, 'iloc') else y_test[:10]
    try:
        test_data_json = sample_test_df.to_json(orient='records')
    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Test verisi JSON dönüşümünde hata: {e}")
        test_data_json = json.dumps([])
    
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Özellikler ölçeklendiriliyor (StandardScaler).")
    # Özellikleri ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Özellikler başarıyla ölçeklendirildi.")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DEBUG: X_train_scaled şekli: {X_train_scaled.shape}, X_test_scaled şekli: {X_test_scaled.shape}")
    
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model seçiliyor: {model_type}")
    # Modeli seç ve eğit
    # Hiperparametreleri model_params'dan al
    current_model_params = {}
    if model_params:
        if model_type == 'random_forest':
            current_model_params['n_estimators'] = int(model_params.get('rf_n_estimators', 100))
            current_model_params['max_depth'] = int(model_params.get('rf_max_depth', 10)) if model_params.get('rf_max_depth') else None
            current_model_params['min_samples_split'] = int(model_params.get('rf_min_samples_split', 2))
            current_model_params['criterion'] = model_params.get('rf_criterion', 'gini')
            logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PARAM: Random Forest parametreleri: {current_model_params}")
            model = RandomForestClassifier(random_state=random_seed, **current_model_params)
        elif model_type == 'svm':
            current_model_params['C'] = float(model_params.get('svm_C', 1.0))
            current_model_params['kernel'] = model_params.get('svm_kernel', 'rbf')
            current_model_params['gamma'] = model_params.get('svm_gamma', 'scale') # 'scale' veya float olabilir
            if current_model_params['gamma'] != 'scale':
                 try:
                    current_model_params['gamma'] = float(current_model_params['gamma'])
                 except ValueError:
                    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: SVM Gamma değeri ('{current_model_params['gamma']}') float'a dönüştürülemedi, 'scale' kullanılacak.")
                    current_model_params['gamma'] = 'scale'

            logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PARAM: SVM parametreleri: {current_model_params}")
            model = SVC(probability=True, random_state=random_seed, **current_model_params)
        elif model_type == 'decision_tree':
            current_model_params['max_depth'] = int(model_params.get('dt_max_depth', 10)) if model_params.get('dt_max_depth') else None
            current_model_params['min_samples_split'] = int(model_params.get('dt_min_samples_split', 2))
            current_model_params['criterion'] = model_params.get('dt_criterion', 'gini')
            logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PARAM: Decision Tree parametreleri: {current_model_params}")
            model = DecisionTreeClassifier(random_state=random_seed, **current_model_params)
        elif model_type == 'naive_bayes':
            current_model_params['var_smoothing'] = float(model_params.get('nb_var_smoothing', 1e-9))
            logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PARAM: Naive Bayes parametreleri: {current_model_params}")
            model = GaussianNB(**current_model_params)
        elif model_type == 'knn':
            current_model_params['n_neighbors'] = int(model_params.get('knn_n_neighbors', 5))
            current_model_params['weights'] = model_params.get('knn_weights', 'uniform')
            current_model_params['metric'] = model_params.get('knn_metric', 'minkowski')
            logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PARAM: KNN parametreleri: {current_model_params}")
            model = KNeighborsClassifier(**current_model_params)
        else: # XGBoost vb. için varsayılan olarak RF kullan
            logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Model tipi '{model_type}' için özel parametreler tanımlanmadı veya model desteklenmiyor. Varsayılan Random Forest parametreleri kullanılacak.")
            model = RandomForestClassifier(n_estimators=100, random_state=random_seed)
    else: # model_params None ise
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Model parametreleri (model_params) sağlanmadı. Varsayılan parametreler kullanılacak.")
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=random_seed)
        elif model_type == 'svm':
                model = SVC(probability=True, random_state=random_seed)
        elif model_type == 'decision_tree':
                model = DecisionTreeClassifier(random_state=random_seed)
        elif model_type == 'naive_bayes':
            model = GaussianNB()
        elif model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Bilinmeyen model tipi: {model_type} ve parametre sağlanmadı.")
            raise ValueError(f"Bilinmeyen model tipi: {model_type}")
        
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model eğitimi başlatılıyor...")
    # Modeli eğit
    model.fit(X_train_scaled, y_train)
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model başarıyla eğitildi.")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DEBUG: Eğitilen modelin parametreleri:\\n{json.dumps(model.get_params(), indent=2)}")
    
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model performansı test verileri üzerinde değerlendiriliyor...")
    # Modelin performansını değerlendir
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    # Precision, recall ve F1 skorlarını hesapla
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) # zero_division eklendi
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)    # zero_division eklendi
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)          # zero_division eklendi
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] METRIC: Doğruluk: {accuracy:.2f}%")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] METRIC: Kesinlik: {precision*100:.2f}%")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] METRIC: Duyarlılık: {recall*100:.2f}%")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] METRIC: F1 Skoru: {f1*100:.2f}%")
    
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Detaylı sınıflandırma raporu oluşturuluyor...")
    try:
        report = classification_report(y_test, y_pred, zero_division=0)
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] REPORT: Sınıflandırma Raporu:\\n{report}")
    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Sınıflandırma raporu oluşturulurken hata: {str(e)}")

    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Karışıklık matrisi oluşturuluyor...")
    try:
        cm = confusion_matrix(y_test, y_pred)
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] MATRIX: Karışıklık Matrisi:\\n{cm}")
    except Exception as e:
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Karışıklık matrisi oluşturulurken hata: {str(e)}")

    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model, ölçekleyici ve özellik adları kaydediliyor...")
    # Modeli ve ölçekleyiciyi kaydet
    model_dir = current_app.config['MODEL_FOLDER']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model klasörü oluşturuldu: {model_dir}")

    model_file_name = f"{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    model_path = os.path.join(model_dir, f"{model_file_name}_model.joblib")
    scaler_path = os.path.join(model_dir, f"{model_file_name}_scaler.joblib")
    feature_path = os.path.join(model_dir, f"{model_file_name}_features.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, feature_path)
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model şuraya kaydedildi: {model_path}")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Ölçekleyici şuraya kaydedildi: {scaler_path}")
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Özellik adları şuraya kaydedildi: {feature_path}")
    
    # Metrikleri sözlüğe ekle
    metrics_dict = {
        'accuracy': accuracy, 
        'precision': precision * 100, 
        'recall': recall * 100, 
        'f1': f1 * 100,
        'train_samples_count': train_samples_count,
        'test_samples_count': test_samples_count,
        'train_data_json': train_data_json,
        'test_data_json': test_data_json,
        'training_logs': logs,  # Logları ekle
        'model_params': current_model_params,  # Model parametrelerini ekle
        'dataset_hash': dataset_hash,  # Veri seti hash değerini ekle
        'random_seed': random_seed  # Random seed değerini ekle
    }
    logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model eğitim süreci tamamlandı.")
    
    return metrics_dict, model_path

def predict_phishing(feature_dict, model_type=None, model_id=None):
    """
    Verilen özellikler kullanılarak, seçili model ile phishing tahmini yapar.
    
    Args:
        feature_dict: Özellikleri içeren sözlük
        model_type: Kullanılacak model tipi ('random_forest', 'svm', 'decision_tree', 'naive_bayes', 'knn')
        model_id: Kullanılacak modelin veritabanındaki ID'si
    
    Returns:
        result: Tahmin sonucu (1: Phishing, 0: Güvenli) ve güven skoru
    """
    if model_id:
        # Eğitilmiş modeli veritabanından bul
        trained_model_entry = TrainedModel.query.get(model_id)
        if not trained_model_entry:
            raise FileNotFoundError(f"Veritabanında ID'si {model_id} olan model bulunamadı.")

        model_path = trained_model_entry.model_path
        # İlgili scaler ve feature dosyalarının yollarını model_path'tan türetmemiz gerekiyor.
        # Kayıt sırasında _model.joblib, _scaler.joblib, _features.joblib konvansiyonunu kullandık.
        base_path = model_path.replace("_model.joblib", "")
        scaler_path = base_path + "_scaler.joblib"
        feature_path = base_path + "_features.joblib"
        selected_model_name = trained_model_entry.model_name # örn: random_forest
        model_display_name = trained_model_entry.model_name_display  # Özel model adı
    elif model_type:
        # Genel model tipini kullan
        model_path = os.path.join(current_app.config['MODEL_FOLDER'], f"{model_type}_model.joblib")
        scaler_path = os.path.join(current_app.config['MODEL_FOLDER'], f"{model_type}_scaler.joblib")
        feature_path = os.path.join(current_app.config['MODEL_FOLDER'], f"{model_type}_features.joblib")
        selected_model_name = model_type
        model_display_name = model_type.upper()
    else:
        raise ValueError("Tahmin için bir model tipi (model_type) veya ID (model_id) belirtilmelidir.")
    
    # Model ve ölçekleyiciyi yükle
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(feature_path)
    except FileNotFoundError as e:
        # Hata mesajını daha açıklayıcı yapalım
        missing_file = "Bilinmiyor"
        if not os.path.exists(model_path):
            missing_file = model_path
        elif not os.path.exists(scaler_path):
            missing_file = scaler_path
        elif not os.path.exists(feature_path):
            missing_file = feature_path
        raise FileNotFoundError(f"Model dosyası ({missing_file}) bulunamadı. Model: {selected_model_name}. Hata: {e}")
    except Exception as e:
        raise Exception(f"Model yüklenirken hata oluştu ({selected_model_name}): {e}")
    
    # Sözlükten özellikleri doğru sırada al
    features = []
    for feature in feature_names:
        if feature in feature_dict:
            features.append(feature_dict[feature])
        else:
            raise ValueError(f"Eksik özellik: {feature}")
    
    # Özellikleri ölçeklendir
    features_scaled = scaler.transform([features])
    
    # Tahmin yap
    prediction = model.predict(features_scaled)[0]
    
    # Güven skorunu hesapla
    if hasattr(model, 'predict_proba'):
        confidence = model.predict_proba(features_scaled)[0][prediction]
    else:
        confidence = None
    
    # Sonucu hazırla
    result = {
        'prediction': int(prediction),
        'prediction_text': 'Phishing' if prediction == 1 else 'Güvenli',
        'confidence': confidence * 100 if confidence is not None else None,
        'model_type': model_display_name
    }
    
    return result
