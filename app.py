from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import pandas as pd
import numpy as np
import joblib
import traceback
import time
from datetime import datetime
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Modeller (XGBoost hariç)
# from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# SQLAlchemy için gerekli importlar
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
# Session için gerekli import
from flask_session import Session
# TrainedModel ve db, init içerisinde tanımlanacak
from app import create_app, db
from app.models import TrainedModel, train_model, predict_phishing, get_model_icon_class

# Uygulama oluştur
app = create_app()

# Session'ı başlat
Session(app)

app.config['SECRET_KEY'] = 'gelistirme-anahtari'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'instance', 'uploads')
app.config['MODEL_FOLDER'] = os.path.join(os.getcwd(), 'instance', 'models')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.getcwd(), 'instance', 'phishing.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True

# Gerekli klasörleri oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), 'flask_session'), exist_ok=True)  # Session dosyaları için klasör

# Session için maksimum yaşam süresi (30 dakika)
app.config['PERMANENT_SESSION_LIFETIME'] = 1800

# Veritabanı tablosunu oluştur
with app.app_context():
    db.create_all()

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Yardımcı Fonksiyonlar
def combine_text_columns(df_or_array):
    """Birden fazla metin sütununu tek bir seride birleştirir."""
    if isinstance(df_or_array, pd.DataFrame):
        return df_or_array.apply(lambda x: ' '.join(x.astype(str)), axis=1)
    elif isinstance(df_or_array, np.ndarray):
        # NumPy dizisi ise (n_samples, n_text_columns)
        return np.apply_along_axis(lambda x: ' '.join(x.astype(str)), 1, df_or_array)
    return df_or_array # Zaten bir Seri veya 1D dizi ise

def flatten_array(arr_2d):
    """(n_samples, 1) şeklindeki bir diziyi (n_samples,) şekline dönüştürür."""
    if hasattr(arr_2d, 'ndim') and arr_2d.ndim == 2 and hasattr(arr_2d, 'shape') and arr_2d.shape[1] == 1:
        return arr_2d.ravel()
    return arr_2d

def detect_column_types(df):
    """
    Veri çerçevesindeki sütunların türlerini tespit eder.
    """
    numeric_cols = []
    categorical_cols = []
    text_cols = []
    
    if not df.columns.any():
        return numeric_cols, categorical_cols, text_cols, None

    target_col = df.columns[-1]
    
    for col in df.columns[:-1]:
        if col == target_col: # Güvenlik önlemi, hedef sütunu atla
            continue

        non_null_series = df[col].dropna()
        if len(non_null_series) == 0: # Tamamen NaN olan sütunlar kategorik olarak ele alınabilir (impute edilecek)
            categorical_cols.append(col)
            print(f"Sütun: {col}, Tümü NaN, kategorik olarak işaretlendi.")
            continue
            
        unique_count = non_null_series.nunique()
        unique_ratio = unique_count / len(non_null_series) if len(non_null_series) > 0 else 0
        
        sample_value_series = non_null_series.head(1)
        sample_value = sample_value_series.iloc[0] if not sample_value_series.empty else None

        print(f"Sütun: {col}, Örnek Değer: {sample_value}, Tip: {type(sample_value)}, Benzersiz Değer Sayısı: {unique_count}, Benzersizlik Oranı: {unique_ratio:.3f}")
        
        # Sayısal tip kontrolü
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            # Eğer çok az benzersiz değer varsa (örn: 0/1 gibi kodlanmış kategorik), kategorik olabilir.
            if unique_count < 5 and unique_count / len(df) < 0.1 : # Örneğin %10'dan az ve 5'ten az farklı değer
                 print(f"Sütun: {col} sayısal ama az benzersiz değere sahip, kategorik olarak değerlendiriliyor.")
                 categorical_cols.append(col)
            else:
                 numeric_cols.append(col)
        # String veya object tip kontrolü
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Ortalama kelime sayısına bakarak metin/kategorik ayrımı
            avg_words = non_null_series.astype(str).apply(lambda x: len(x.split())).mean()
            if avg_words > 3.0 and unique_ratio > 0.5 : # Ortalama 3 kelimeden fazla ve yüksek benzersizlik -> metin
                text_cols.append(col)
            elif unique_count <= 20 or unique_ratio < 0.1: # Az benzersiz değer veya düşük oran -> kategorik
                categorical_cols.append(col)
            else: # Arada kalanlar, daha çok metne yakınsa metin, değilse kategorik
                if avg_words > 1.5 and unique_ratio > 0.2:
                     text_cols.append(col)
                else:
                     categorical_cols.append(col)
        # Boolean tip kontrolü (kategorik olarak ele al)
        elif pd.api.types.is_bool_dtype(df[col]):
            categorical_cols.append(col)
        # Diğer tipler (tarih vs. olabilir, şimdilik kategorik)
        else:
            if unique_count < len(df) * 0.1 : # Benzersiz değerler toplam satır sayısının %10'undan azsa kategorik
                categorical_cols.append(col)
            else: # Yüksek ihtimalle ID gibi bir sütun veya işlenmesi gereken farklı bir tip, şimdilik metin
                # Bu tür sütunlar genellikle atılır veya özel işlenir.
                # print(f"Sütun: {col} tipi {df[col].dtype} belirsiz, metin olarak işaretleniyor ama kontrol edilmeli.")
                # text_cols.append(col) # Ya da ID gibi ise atmak daha iyi olabilir
                # Şimdilik kategorik olarak işaretleyelim ki en azından bir işleme tabi tutulsun
                 categorical_cols.append(col)


    print(f"Tespit edilen sütun tipleri:\n  Sayısal: {numeric_cols}\n  Kategorik: {categorical_cols}\n  Metin: {text_cols}\n  Hedef: {target_col}")
    return numeric_cols, categorical_cols, text_cols, target_col

def ensure_str(value):
    if pd.isna(value):
        return "" # NaN değerleri boş string yapar
    if isinstance(value, (list, tuple)): # Liste veya tuple ise elemanları birleştir
        return ' '.join(map(ensure_str, value))
    if isinstance(value, float) and value.is_integer(): # 1.0 gibi float'ları '1' yapar
        return str(int(value))
    return str(value) # Diğer her şeyi string'e çevirir


def preprocess_dataframe(df):
    df = df.dropna(how='all').copy() # .copy() ile SettingWithCopyWarning önlenir
    for col in df.columns:
        # Sayısal olması beklenen ama object olanları sayısala çevirmeyi dene
        if df[col].dtype == 'object':
            try:
                # Sadece gerçekten sayısal olabilecekleri çevir, karışık tipleri bozma
                converted_col = pd.to_numeric(df[col], errors='raise')
                # Eğer çok az NaN oluştuysa ve orijinalde sayılar varsa, bu iyi bir işaret olabilir
                if converted_col.isna().sum() < len(df) * 0.1:
                    df[col] = converted_col
                    print(f"Sütun '{col}' (object) başarıyla sayısala dönüştürüldü.")
            except (ValueError, TypeError):
                # Sayısala çevrilemiyorsa, string olarak işle
                df[col] = df[col].apply(ensure_str)
        
        # Kalan object veya string tipleri için ensure_str uygula (sayısala dönenler hariç)
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].apply(ensure_str)
            df[col] = df[col].fillna('') # ensure_str sonrası NaN kalırsa (pek olası değil)
        elif pd.api.types.is_numeric_dtype(df[col]):
             # Sayısal sütunlardaki NaN'ları özel bir değerle değil, olduğu gibi bırakalım.
             # Imputer'lar bu NaN'ları daha sonra işleyecek.
             pass # df[col] = df[col].fillna(np.nan) # veya doğrudan bırak
        else: # Diğer tipler (bool, datetime vs.)
             df[col] = df[col].apply(ensure_str).fillna('')
    return df

def train_model(file_path, model_type='random_forest'):
    try:
        print(f"CSV dosyası yükleniyor: {file_path}")
        try:
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
        
        print(f"CSV yüklendi. Orijinal Satır sayısı: {len(df)}, Sütun sayısı: {len(df.columns)}")
        
        # Kullanıcıya ilk birkaç satırı ve tipleri gösterelim
        print("Veri setinin ilk 5 satırı (orijinal):")
        print(df.head().to_string())
        print("Orijinal Sütun Tipleri:")
        print(df.dtypes)

        df_processed = preprocess_dataframe(df.copy()) # Orijinal df'yi bozmamak için kopya alalım
        
        print("Ön işlenmiş veri setinin ilk 5 satırı:")
        print(df_processed.head().to_string())
        print("Ön İşlenmiş Sütun Tipleri:")
        print(df_processed.dtypes)

        numeric_cols, categorical_cols, text_cols, target_col = detect_column_types(df_processed.copy())
        
        if not target_col:
            raise ValueError("Hedef sütun tespit edilemedi (CSV'nin son sütunu olmalı).")
        if target_col not in df_processed.columns:
            raise ValueError(f"Hedef sütun '{target_col}' veri çerçevesinde bulunamadı.")

        y_series = df_processed[target_col]
        le = LabelEncoder()
        y = le.fit_transform(y_series.astype(str))
        target_classes = le.classes_
        print(f"Hedef sütun '{target_col}' etiketlendi. Sınıflar: {target_classes}, Kodlanmış y değerleri: {np.unique(y)}")
        
        X = df_processed.drop(columns=[target_col])
        # Stratify=y, eğer y'de en az 2 sınıf ve her sınıfta en az 2 örnek varsa çalışır (cv için de geçerli)
        # np.unique(y).size >= 2 and all(np.bincount(y) >= 2) # gibi bir kontrol yapılabilir
        min_samples_per_class_for_stratify = 2 # Genellikle cv değeri kadar olmalı
        can_stratify = np.unique(y).size >= 2 and all(np.bincount(y)[np.unique(y, return_inverse=True)[1]] >= min_samples_per_class_for_stratify)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None)
        
        # Eğitim ve test veri sayılarını ve örneklerini saklama
        train_samples_count = len(X_train)
        test_samples_count = len(X_test)
        
        print(f"DEBUG: Eğitim veri seti boyutu: {train_samples_count}, Test veri seti boyutu: {test_samples_count}")
        
        # Eğitim ve test verilerinden örnekleri JSON formatında sakla (en fazla 10 örnek)
        # Eğitim verisi örnekleri - hedef değerleriyle birlikte
        sample_train_df = pd.DataFrame(X_train.iloc[:10].copy())
        sample_train_df[target_col] = [target_classes[y] for y in y_train[:10]]
        train_data_json = sample_train_df.to_json(orient='records')
        
        # Test verisi örnekleri - hedef değerleriyle birlikte
        sample_test_df = pd.DataFrame(X_test.iloc[:10].copy())
        sample_test_df[target_col] = [target_classes[y] for y in y_test[:10]]
        test_data_json = sample_test_df.to_json(orient='records')
        
        print(f"DEBUG: JSON veri örneği mevcut: train_data={bool(train_data_json)}, test_data={bool(test_data_json)}")
        
        preprocessor_steps = []
        
        if numeric_cols:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')), # NaN'ları ortalama ile doldur
                ('scaler', StandardScaler())
            ])
            preprocessor_steps.append(('num', numeric_transformer, numeric_cols))
        
        if categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='_MISSING_')), # NaN'ları veya boş stringleri özel bir etiketle doldur
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
            ])
            preprocessor_steps.append(('cat', categorical_transformer, categorical_cols))
        
        if text_cols:
            text_combiner_transformer = FunctionTransformer(combine_text_columns, validate=False)
            # Metin sütunlarındaki boş stringler (önceden NaN olanlar) imputer ile tekrar işlenir.
            # TfidfVectorizer zaten boş stringleri doğru şekilde ele alır.
            imputer_for_text = SimpleImputer(strategy='constant', fill_value='') # Aslında gerekmeyebilir, Tfidf boş stringleri atlar.
            array_flattener_transformer = FunctionTransformer(flatten_array, validate=False)

            text_processing_pipeline = Pipeline(steps=[
                ('combiner', text_combiner_transformer), # Birden fazla metin sütununu birleştirir
                 # ('imputer', imputer_for_text), # Bu adım Tfidf'den önce genellikle gereksizdir
                ('flattener', array_flattener_transformer), # (n,1) -> (n,) yapar
                ('tfidf', TfidfVectorizer(max_features=1000, min_df=2, ngram_range=(1,2), stop_words=None)) # stop_words='english' eklenebilir
            ])
            preprocessor_steps.append(('text', text_processing_pipeline, text_cols))
        
        if not preprocessor_steps:
            # Eğer hiç özellik sütunu kalmadıysa (örn: sadece hedef sütun varsa veya tüm özellikler atıldıysa)
            # Bu durumda model eğitilemez.
            # Ya da, bu durumu ele almak için DummyClassifier gibi bir şey kullanılabilir.
            # Şimdilik hata verelim.
            features_available = X.columns.tolist()
            raise ValueError(f"İşlenecek özellik sütunu bulunamadı (sayısal, kategorik veya metin). "
                             f"Kullanılabilir özellikler: {features_available}. "
                             f"Hedef sütun: {target_col}. "
                             f"Lütfen veri setinizi ve sütun tipi tespitini kontrol edin.")


        preprocessor = ColumnTransformer(
            transformers=preprocessor_steps,
            remainder='drop', # Geriye kalan sütunları at (örn: ID sütunları)
            sparse_threshold=0 # Yoğun (dense) çıktı almayı dene, Naive Bayes gibi bazı modeller için önemli
        )
        
        # Model parametreleri
        param_grids = {
            'random_forest': {
                'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10],
                'classifier__min_samples_split': [2, 5]
            },
            'svm': { # SVM için daha az parametre, daha hızlı deneme
                'classifier__C': [0.1, 1], 'classifier__kernel': ['linear'] 
            },
            'decision_tree': {
                'classifier__max_depth': [None, 10], 'classifier__min_samples_split': [2, 5],
                'classifier__criterion': ['gini', 'entropy']
            },
            'naive_bayes': { # MultinomialNB için (TFIDF sonrası)
                'classifier__alpha': [0.1, 0.5, 1.0]
            },
            'knn': {
                'classifier__n_neighbors': [3, 5], 'classifier__weights': ['uniform', 'distance'],
                # 'classifier__metric': ['euclidean', 'manhattan'] # Çok fazla kombinasyon
            }
        }

        # Model seçimi
        if model_type == 'random_forest': classifier = RandomForestClassifier(random_state=42)
        elif model_type == 'svm': classifier = SVC(probability=True, random_state=42)
        elif model_type == 'decision_tree': classifier = DecisionTreeClassifier(random_state=42)
        elif model_type == 'naive_bayes':
            # Eğer metin özelliği yoksa GaussianNB kullanılabilir, ama preprocessor sonrası veri seyrekse sorun çıkarır.
            # ColumnTransformer'da sparse_threshold=0 ayarı ile yoğun çıktı almaya çalışıyoruz.
            # if not text_cols and numeric_cols: # Basit bir mantık
            #     classifier = GaussianNB()
            #     param_grids['naive_bayes'] = {'classifier__var_smoothing': [1e-9, 1e-8]}
            # else: # Metin varsa veya kategorik ağırlıklıysa MultinomialNB daha iyi olabilir (yoğun veri üzerinde de çalışır)
            classifier = MultinomialNB() # TFIDF seyrek olsa bile MultinomialNB çalışır
        elif model_type == 'knn': classifier = KNeighborsClassifier()
        else: raise ValueError(f"Bilinmeyen model tipi: {model_type}")
        
        current_param_grid = param_grids[model_type]

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        print(f"Hiperparametre optimizasyonu ({model_type}) için CV=3 ile başlıyor...")
        cv_folds = 3 # Hızlı denemeler için 3, daha güvenilir sonuç için 5
        if can_stratify and all(np.bincount(y)[np.unique(y, return_inverse=True)[1]] >= cv_folds):
            pass # Stratify kullanılabilir
        else:
            print(f"Uyarı: Sınıf dağılımı nedeniyle {cv_folds}-katlı stratifikasyon yapılamıyor, normal CV kullanılacak.")
            can_stratify = False # Stratifikasyonu devre dışı bırak


        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=current_param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            error_score='raise'
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"En iyi parametreler: {best_params}")
        
        # Cross-validation skorları
        # En iyi pipeline ile tüm X, y üzerinde CV yapalım
        if can_stratify and all(np.bincount(y)[np.unique(y, return_inverse=True)[1]] >= cv_folds) :
             cv_obj_for_final_score = cv_folds
        else:
             cv_obj_for_final_score = cv_folds # Stratify olmadan devam et

        cv_scores = cross_val_score(best_model, X, y, cv=cv_obj_for_final_score, scoring='accuracy')
        cv_mean = np.mean(cv_scores) * 100
        cv_std = np.std(cv_scores) * 100
        print(f"{cv_folds}-katlı CV doğruluk (en iyi model ile): {cv_mean:.2f}% (±{cv_std:.2f})")
        
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Test seti doğruluk: {accuracy:.2f}%, Kesinlik: {precision:.4f}, Duyarlılık: {recall:.4f}, F1: {f1:.4f}")
        print(f"Karmaşıklık Matrisi:\n{conf_matrix}")
        
        class_report_str = classification_report(y_test, y_pred, target_names=target_classes.astype(str), zero_division=0)
        print(f"Sınıflandırma Raporu:\n{class_report_str}")
        
        model_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_type}_model.joblib")
        meta_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_type}_meta.joblib")
        
        model_meta = {
            'numeric_columns': numeric_cols, 'categorical_columns': categorical_cols, 'text_columns': text_cols,
            'target_col': target_col, 'target_classes': target_classes.tolist(),
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
            'cv_mean': cv_mean, 'cv_std': cv_std, 'best_params': best_params,
            'features_in_order': X.columns.tolist(), # Pipeline'ın beklediği sütun sırası
            'class_report_str': class_report_str
        }
        
        joblib.dump(best_model, model_path)
        joblib.dump(model_meta, meta_path)
        
        metrics_dict = {
            'accuracy': accuracy, 
            'precision': precision * 100, 
            'recall': recall * 100, 
            'f1': f1 * 100,
            'cv_mean': cv_mean, 
            'cv_std': cv_std, 
            'best_params': best_params,
            'class_report_str': class_report_str,
            'train_samples_count': train_samples_count,
            'test_samples_count': test_samples_count,
            'train_data_json': train_data_json,
            'test_data_json': test_data_json
        }
        
        # Debug bilgileri yazdır
        print(f"DEBUG metrics_dict içeriği: {metrics_dict.keys()}")
        
        # Model sonuçlarını oluştur
        model_result = {
            'model_type': model_type,
            'model_name': model_type.upper(),
            'model_name_display': model_type.upper(),  # Gösterilecek model adı
            'dataset_name': file_path.split('/')[-1],
            'accuracy': metrics_dict['accuracy'],
            'precision': metrics_dict['precision'],
            'recall': metrics_dict['recall'],
            'f1_score': metrics_dict['f1'],
            'training_time': time.time() - start_time,
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            'icon_class': get_model_icon_class(model_type),
            'train_samples_count': metrics_dict.get('train_samples_count', 0),
            'test_samples_count': metrics_dict.get('test_samples_count', 0),
            'train_data_json': metrics_dict.get('train_data_json', '[]'),
            'test_data_json': metrics_dict.get('test_data_json', '[]')
        }
        
        # Başarılı eğitim için flash mesajı
        flash(f'{model_type.upper()} modeli başarıyla eğitildi.', 'success')
        
        # Sonuçları oturum değişkeninde sakla
        model_result_list = [model_result]
        session['training_results'] = model_result_list
        
        # Debug bilgileri
        print(f"DEBUG: model_result içeriği: {model_result.keys()}")
        print(f"DEBUG: train_samples_count={model_result.get('train_samples_count')}, test_samples_count={model_result.get('test_samples_count')}")
        print(f"DEBUG: train_data_json uzunluğu: {len(model_result.get('train_data_json', ''))}")
        print(f"DEBUG: test_data_json uzunluğu: {len(model_result.get('test_data_json', ''))}")
        
        # Kaydetme sayfasını göster
        return metrics_dict, model_path
    
    except ValueError as ve: # Özellikle beklenen hatalar
        print(f"Model eğitimi sırasında Değer Hatası: {str(ve)}")
        traceback.print_exc()
        raise ve # Hata mesajını flash'ta göstermek için tekrar raise et
    except Exception as e:
        print(f"Model eğitimi sırasında genel bir hata oluştu: {str(e)}")
        traceback.print_exc()
        raise RuntimeError(f"Model eğitiminde beklenmedik bir sorun oluştu: {type(e).__name__} - {e}")


def predict_phishing(feature_dict, model_type='random_forest'):
    model_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_type}_model.joblib")
    meta_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_type}_meta.joblib")
    
    try:
        model = joblib.load(model_path)
        model_meta = joblib.load(meta_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"'{model_type}' modeli veya meta dosyası bulunamadı. Lütfen önce modeli eğitin.")
    
    # Özellikleri modelin eğitildiği sıraya göre DataFrame'e hazırla
    features_in_order = model_meta.get('features_in_order', [])
    if not features_in_order:
        # Eski modellerde bu bilgi olmayabilir, feature_dict'ten gelen sırayı kullan
        # Ancak bu ColumnTransformer için sorun yaratabilir. En iyisi modelin bilmesi.
        # Ya da tüm olası sütunları alıp, olmayanları varsayılanla doldur.
        # Şimdilik basitçe gelen feature_dict'in anahtarlarını alalım
        print("Uyarı: Model meta verisinde 'features_in_order' bulunamadı. Gelen özellik sırası kullanılıyor.")
        # features_in_order = list(feature_dict.keys()) # Bu riskli
        # Daha güvenli bir yaklaşım: tüm olası sütunları (numeric, cat, text) birleştirmek
        all_known_cols = model_meta.get('numeric_columns', []) + \
                         model_meta.get('categorical_columns', []) + \
                         model_meta.get('text_columns', [])
        # Eğer features_in_order hala boşsa, bu sütunları kullan
        if not all_known_cols:
             raise ValueError("Model meta verisinde özellik listesi bulunamadı.")
        # Gelen feature_dict'teki anahtarların bir alt kümesi olabilir.
        # Bizim için önemli olan, modelin `predict` metoduna giden DataFrame'in
        # `ColumnTransformer`'ın `fit` sırasında gördüğü sütunlara sahip olması.
        # Bu yüzden `features_in_order` meta veride kritik.

        # Güvenli liman: Eğer features_in_order yoksa, hata ver.
        # Bu, modelin doğru şekilde kaydedilmediğini gösterir.
        raise ValueError("Model meta verisi eksik: 'features_in_order' anahtarı bulunamadı. Modelin yeniden eğitilmesi gerekebilir.")


    # Gelen feature_dict'i DataFrame'e dönüştür
    # Eksik sütunları NaN ile doldur, pipeline'daki imputer'lar halleder
    # Sütun sırası 'features_in_order'a göre olmalı
    df_input_dict = {col: [feature_dict.get(col)] for col in features_in_order}
    feature_df = pd.DataFrame.from_dict(df_input_dict)
    
    # Veri tiplerini de ayarlamak gerekebilir, özellikle predict için.
    # Model meta'dan tipleri alıp uygulamak daha iyi olur.
    # Şimdilik, ColumnTransformer'ın bunu halledeceğini varsayalım,
    # yeter ki sütun isimleri ve sırası doğru olsun.
    # Gelen değerleri string olarak alıp, pipeline'ın işlemesine bırakmak daha güvenli olabilir.
    for col in features_in_order:
        if col in feature_dict: # Sadece formdan gelenleri işle
            # Gelen değeri string'e çevir, pipeline içindeki numeric/cat/text işlemleri halleder
            feature_df[col] = feature_df[col].astype(str).map(ensure_str) 
            # Sayısal olması beklenenler için özel bir durum
            if col in model_meta.get('numeric_columns', []):
                # Sayısala çevirmeyi dene, olmazsa NaN bırak (imputer halleder)
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
        else: # Formdan gelmeyen özellikler için NaN ata (imputer halleder)
             feature_df[col] = np.nan


    print(f"Tahmin için hazırlanan DataFrame (ilk satır):\n{feature_df.head(1).to_string()}")

    try:
        prediction_numeric_array = model.predict(feature_df)
        prediction_numeric = prediction_numeric_array[0]
        
        target_classes = model_meta.get('target_classes', ['0', '1'])
        prediction_label = target_classes[prediction_numeric] # Dönüşüm LabelEncoder ile yapıldığı için indexleme
        
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_df)[0]
            confidence = proba[prediction_numeric] # En olası sınıfın olasılığı
        
        # prediction_text'i daha genel yapalım
        # Örneğin, target_classes ['Legitimate', 'Phishing'] ise prediction_label doğrudan kullanılabilir.
        # Eğer [0, 1] ise ve 1 Phishing demekse, ona göre çeviri yapılır.
        # Şimdilik basit bir mantık: eğer label "1" veya (string) "Phishing" içeriyorsa Phishing kabul et.
        # Bu, target_classes'ın içeriğine bağlı.
        pred_label_str = str(prediction_label).lower()
        positive_class_label_str = str(target_classes[-1]).lower() # Genelde son sınıf pozitif olur (örn: Phishing)

        if pred_label_str == positive_class_label_str or pred_label_str == "phishing" or pred_label_str == "1":
            prediction_text = "Phishing" # Ya da prediction_label'ı doğrudan kullan
        else:
            prediction_text = "Güvenli" # Ya da prediction_label

    except Exception as e:
        print(f"Tahmin sırasında hata: {str(e)}")
        traceback.print_exc()
        print(f"Girdi Verisi (DataFrame):\n{feature_df.to_string()}")
        raise RuntimeError(f"Tahmin yapılırken bir sorun oluştu: {type(e).__name__} - {e}")
    
    result = {
        'prediction': int(prediction_numeric),
        'prediction_label': str(prediction_label),
        'prediction_text': prediction_text,
        'confidence': confidence * 100 if confidence is not None else None,
        'model_type': model_type.upper()
    }
    return result

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about_route():
    return render_template('about.html')

@app.route('/train', methods=['GET', 'POST'])
def train_route():
    # Oturumun kalıcı olmasını sağla
    session.permanent = True
    
    if request.method == 'POST':
        # Formdan model kaydetme kararı
        if 'save_model' in request.form and request.form.get('save_decision') == 'yes':
            try:
                model_name_display = request.form.get('model_name', '').strip()
                if not model_name_display:
                    flash('Eğitim adı boş bırakılamaz!', 'danger')
                    # Sonuçları ve girilen diğer bilgileri kullanarak save_model sayfasını tekrar render et
                    # Bu, session'dan son sonuçları almayı gerektirir
                    last_result_for_save = session.get('last_result_for_save')
                    if last_result_for_save:
                        return render_template('save_model.html', results=[last_result_for_save], trained_models=TrainedModel.query.order_by(TrainedModel.trained_at.desc()).all(), model_name_display=model_name_display)
                    else:
                        return redirect(url_for('train_route'))


                # Aynı isimde model olup olmadığını kontrol et
                existing_model_check = TrainedModel.query.filter_by(model_name_display=model_name_display).first()
                if existing_model_check:
                    flash(f"'{model_name_display}' adında bir eğitim zaten mevcut! Lütfen farklı bir isim girin veya mevcut kaydı silin.", 'danger')
                    last_result_for_save = session.get('last_result_for_save')
                    if last_result_for_save:
                         # Kullanıcıya girdiği ismi ve hatayı göstermek için model_name_display'i de gönderiyoruz
                        return render_template('save_model.html', results=[last_result_for_save], trained_models=TrainedModel.query.order_by(TrainedModel.trained_at.desc()).all(), model_name_display_input=model_name_display)
                    else:
                        # Eğer session'da sonuç yoksa (beklenmedik durum), ana eğitim sayfasına yönlendir
                        return redirect(url_for('train_route'))
                
                model_type = request.form.get('model_type')
                dataset_name = request.form.get('dataset_name')
                accuracy = float(request.form.get('accuracy'))
                precision = float(request.form.get('precision'))
                recall = float(request.form.get('recall'))
                f1_score_val = float(request.form.get('f1_score'))
                training_time = float(request.form.get('training_time'))
                model_path = request.form.get('model_path')
                
                train_samples_count = request.form.get('train_samples_count', type=int)
                test_samples_count = request.form.get('test_samples_count', type=int)
                train_data_json = request.form.get('train_data_json', '{}')
                test_data_json = request.form.get('test_data_json', '{}')
                
                new_trained_model = TrainedModel(
                    model_name=model_type,
                    model_name_display=model_name_display,
                    dataset_name=dataset_name,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score_val,
                    training_time=training_time,
                    model_path=model_path,
                    icon_class=get_model_icon_class(model_type),
                    train_samples_count=train_samples_count,
                    test_samples_count=test_samples_count,
                    training_data_json=train_data_json,
                    test_data_json=test_data_json
                )
                db.session.add(new_trained_model)
                db.session.commit()
                flash(f'{model_name_display} modeli başarıyla kaydedildi!', 'success')
                session.pop('last_result_for_save', None) # Kayıt sonrası session'ı temizle
                return redirect(url_for('train_route'))
            except IntegrityError: # sqlalchemy.exc.IntegrityError
                db.session.rollback()
                flash(f'HATA: \'{model_name_display}\' adında bir eğitim zaten mevcut! Lütfen farklı bir isim seçin.', 'danger')
                last_result_for_save = session.get('last_result_for_save')
                if last_result_for_save:
                    return render_template('save_model.html', results=[last_result_for_save], trained_models=TrainedModel.query.order_by(TrainedModel.trained_at.desc()).all(), model_name_display_input=model_name_display)
                return redirect(url_for('train_route')) # Fallback
            except Exception as e:
                db.session.rollback()
                flash(f'Model kaydedilirken hata oluştu: {str(e)}', 'danger')
                app.logger.error(f"Model kaydetme hatası: {traceback.format_exc()}")
                # Hata durumunda, mümkünse save_model sayfasına sonuçlarla geri dön
                last_result_for_save = session.get('last_result_for_save')
                if last_result_for_save:
                    return render_template('save_model.html', results=[last_result_for_save], trained_models=TrainedModel.query.order_by(TrainedModel.trained_at.desc()).all())
                return redirect(url_for('train_route'))

        elif 'save_model' in request.form and request.form.get('save_decision') == 'no':
            flash('Model kaydedilmedi.', 'info')
            session.pop('last_result_for_save', None) # Kayıttan vazgeçilince de session'ı temizle
            return redirect(url_for('train_route'))
        
        # Dosya yükleme ve model eğitimi
        if 'file' not in request.files:
            flash('Dosya seçilmedi!', 'warning')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Dosya seçilmedi!', 'warning')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(file_path)
                model_type = request.form.get('model_type', 'random_forest')
                
                # Model parametrelerini formdan al
                model_params = {}
                if model_type == 'random_forest':
                    model_params['n_estimators'] = request.form.get('rf_n_estimators', default=100, type=int)
                    model_params['max_depth'] = request.form.get('rf_max_depth', default=None, type=int)
                    model_params['min_samples_split'] = request.form.get('rf_min_samples_split', default=2, type=int)
                    model_params['criterion'] = request.form.get('rf_criterion', default='gini', type=str)
                elif model_type == 'svm':
                    model_params['C'] = request.form.get('svm_C', default=1.0, type=float)
                    model_params['kernel'] = request.form.get('svm_kernel', default='rbf', type=str)
                    model_params['gamma'] = request.form.get('svm_gamma', default='scale', type=str) # 'auto' veya float olabilir
                    if model_params['gamma'] not in ['scale', 'auto']:
                        try:
                            model_params['gamma'] = float(model_params['gamma'])
                        except ValueError:
                            model_params['gamma'] = 'scale' # Geçersizse varsayılana dön
                elif model_type == 'decision_tree':
                    model_params['max_depth'] = request.form.get('dt_max_depth', default=None, type=int)
                    model_params['min_samples_split'] = request.form.get('dt_min_samples_split', default=2, type=int)
                    model_params['criterion'] = request.form.get('dt_criterion', default='gini', type=str)
                elif model_type == 'naive_bayes':
                    model_params['var_smoothing'] = request.form.get('nb_var_smoothing', default=1e-9, type=float)
                elif model_type == 'knn':
                    model_params['n_neighbors'] = request.form.get('knn_n_neighbors', default=5, type=int)
                    model_params['weights'] = request.form.get('knn_weights', default='uniform', type=str)
                    model_params['metric'] = request.form.get('knn_metric', default='minkowski', type=str)
                
                # Veri bölme oranını formdan al
                train_test_split_ratio = request.form.get('train_test_split', default=80, type=int) / 100.0


                results = train_model(file_path, model_type, model_params, train_test_split_ratio)
                
                results_list = [{
                    'model_type': model_type,
                    'model_name': model_type.replace('_', ' ').title(),
                    'model_name_display': f"{model_type.replace('_', ' ').title()} ({os.path.splitext(filename)[0]})", # Kayıt için önerilen isim
                    'dataset_name': filename,
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1'],
                    'training_time': results['training_time'],
                    'model_path': results['model_path'],
                    'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'icon_class': get_model_icon_class(model_type),
                    'train_samples_count': results.get('train_samples_count'),
                    'test_samples_count': results.get('test_samples_count'),
                    'train_data_json': results.get('train_data_json', '{}'),
                    'test_data_json': results.get('test_data_json', '{}')
                }]
                
                # Sonucu session'da sakla (save_model sayfasında kullanmak ve hata durumunda geri dönmek için)
                if results_list:
                    session['last_result_for_save'] = results_list[0]
                
                trained_models = TrainedModel.query.order_by(TrainedModel.trained_at.desc()).all()
                return render_template('save_model.html', results=results_list, trained_models=trained_models)

            except ValueError as ve:
                flash(f'Model eğitimi sırasında veri hatası: {str(ve)}', 'danger')
                app.logger.error(f"Veri Hatası: {traceback.format_exc()}")
            except RuntimeError as re:
                 flash(f'Model eğitimi sırasında çalışma zamanı hatası: {str(re)}', 'danger')
                 app.logger.error(f"Runtime Hatası: {traceback.format_exc()}")
            except Exception as e:
                flash(f'Model eğitimi sırasında beklenmedik bir hata oluştu: {str(e)}', 'danger')
                app.logger.error(f"Beklenmedik Hata: {traceback.format_exc()}")
            
            # Hata durumunda ana eğitim sayfasına yönlendir
            return redirect(url_for('train_route'))
        else:
            flash('İzin verilmeyen dosya türü!', 'danger')
            return redirect(request.url)
    
    # GET isteği
    trained_models = TrainedModel.query.order_by(TrainedModel.trained_at.desc()).all()
    # Sonuçları session'dan temizle (yeni bir eğitim için)
    session.pop('last_results_for_save_page', None) 
    session.pop('last_result_for_save', None)
    return render_template('train.html', trained_models=trained_models, results=None)

@app.route('/delete_history_entry/<int:entry_id>', methods=['POST'])
def delete_history_entry_route(entry_id):
    model = TrainedModel.query.get_or_404(entry_id)
    
    db.session.delete(model)
    db.session.commit()
    
    flash('Eğitim kaydı silindi', 'success')
    return redirect(url_for('train_route'))

@app.route('/delete_all_history', methods=['POST'])
def delete_all_history_route():
    models = TrainedModel.query.all()
    
    for model in models:
        db.session.delete(model)
    
    db.session.commit()
    
    flash('Tüm eğitim geçmişi silindi', 'success')
    return redirect(url_for('train_route'))

@app.route('/test', methods=['GET', 'POST'])
def test_route():
    model_types = ['random_forest', 'svm', 'decision_tree', 'naive_bayes', 'knn']
    # Eğitilmiş modelleri veritabanından çek
    trained_models = TrainedModel.query.order_by(TrainedModel.trained_at.desc()).all()
    
    # Debug: Veritabanından çekilen modelleri loglama
    print(f"Veritabanından çekilen eğitilmiş model sayısı: {len(trained_models)}")
    for model in trained_models:
        print(f"Model ID: {model.id}, Tip: {model.model_name}, Ad: {model.model_name_display}, Dosya: {model.model_path}")
    
    # Test sayfasında gösterilecek varsayılan özellikler
    current_model_type_for_features = request.form.get('model_type', model_types[0]) 
    form_features_dict = {}
    try:
        # POST varsa form verilerini kullan, yoksa GET parametrelerine bak
        selected_model_id = request.form.get('trained_model_id', None)
        
        if selected_model_id:
            # Eğitilmiş model ID'si varsa, o modelin özelliklerini yükle
            trained_model = TrainedModel.query.get(selected_model_id)
            if trained_model:
                # Buraya önemli değişiklik: kayıtlı modelin meta dosyasını veya feature dosyasını bul
                model_path = trained_model.model_path
                model_base_path = model_path.replace("_model.joblib", "")
                possible_meta_path = model_base_path + "_meta.joblib"
                possible_feature_path = model_base_path + "_features.joblib"
                
                if os.path.exists(possible_meta_path):
                    model_meta_for_form = joblib.load(possible_meta_path)
                    all_features_from_meta = model_meta_for_form.get('features_in_order', [])
                elif os.path.exists(possible_feature_path):
                    # Eğer meta dosyası yoksa feature dosyasını kullan
                    all_features_from_meta = joblib.load(possible_feature_path)
                else:
                    # Model özel dosyaları yoksa, model tipinin genel meta bilgilerine bak
                    meta_path_from_type = os.path.join(app.config['MODEL_FOLDER'], f"{trained_model.model_name}_meta.joblib")
                    if os.path.exists(meta_path_from_type):
                        model_meta_for_form = joblib.load(meta_path_from_type)
                        all_features_from_meta = model_meta_for_form.get('features_in_order', [])
                    else:
                        all_features_from_meta = []
                        flash(f"'{trained_model.model_name_display}' modelinin özellik listesi bulunamadı.", 'warning')
                
                for feature_name in all_features_from_meta:
                    form_features_dict[feature_name] = {'label': feature_name.replace('_', ' ').title(), 'type': 'text', 'default': ''}
                
                # Seçilen modele göre model_type değerini güncelle
                current_model_type_for_features = trained_model.model_name
            else:
                flash(f"ID: {selected_model_id} olan model bulunamadı.", 'warning')
        else:
            # Model tipi seçildiyse, onun meta dosyasını kullan
            meta_path_for_form = os.path.join(app.config['MODEL_FOLDER'], f"{current_model_type_for_features}_meta.joblib")
        if os.path.exists(meta_path_for_form):
            model_meta_for_form = joblib.load(meta_path_for_form)
            all_features_from_meta = model_meta_for_form.get('features_in_order', [])
            for feature_name in all_features_from_meta:
                form_features_dict[feature_name] = {'label': feature_name.replace('_', ' ').title(), 'type': 'text', 'default': ''}
        else:
             flash(f"'{current_model_type_for_features}' için özellik listesi yüklenemedi. Lütfen önce modeli eğitin.", "info")
    except Exception as e:
        print(f"Test formu için özellikler yüklenirken hata: {e}")
        print(traceback.format_exc())

    if request.method == 'POST':
        model_type_selected = request.form.get('model_type', model_types[0])
        trained_model_id = request.form.get('trained_model_id')
        
        # Formdan gelen tüm anahtarları al
        features_to_predict = {}
        try:
            # Eğer eğitilmiş bir model seçildiyse, o modelin özelliklerini kullan
            if trained_model_id:
                trained_model = TrainedModel.query.get(trained_model_id)
                if not trained_model:
                    flash(f"ID: {trained_model_id} olan model bulunamadı.", 'danger')
                    return render_template('test.html', model_types=model_types, current_model_type=model_type_selected, 
                                         form_features_dict=form_features_dict, trained_models=trained_models, submitted_features={})
                
                # Özellik listesini bul - önce modelin kendi dosyalarına, sonra genel meta dosyasına bak
                model_path = trained_model.model_path
                model_base_path = model_path.replace("_model.joblib", "")
                meta_path = model_base_path + "_meta.joblib"
                feature_path = model_base_path + "_features.joblib"
                
                if os.path.exists(meta_path):
                    model_meta = joblib.load(meta_path)
                    expected_feature_names = model_meta.get('features_in_order', [])
                elif os.path.exists(feature_path):
                    expected_feature_names = joblib.load(feature_path)
                else:
                    # Özel dosyalar bulunamazsa, genel model tipinin meta dosyasını kullan
                    meta_path = os.path.join(app.config['MODEL_FOLDER'], f"{trained_model.model_name}_meta.joblib")
                    if os.path.exists(meta_path):
                        model_meta = joblib.load(meta_path)
                        expected_feature_names = model_meta.get('features_in_order', [])
                    else:
                        flash(f"'{trained_model.model_name_display}' modelinin özellik listesi bulunamadı.", 'danger')
                        return render_template('test.html', model_types=model_types, current_model_type=model_type_selected, 
                                             form_features_dict=form_features_dict, trained_models=trained_models, submitted_features={})
            else:
                # Genel model tipinin özelliklerini kullan
                meta_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_type_selected}_meta.joblib")
                if not os.path.exists(meta_path):
                    flash(f"'{model_type_selected}' için meta dosyası bulunamadı. Lütfen önce modeli eğitin.", 'danger')
                    return render_template('test.html', model_types=model_types, current_model_type=model_type_selected, 
                                         form_features_dict=form_features_dict, trained_models=trained_models, submitted_features={})
            model_meta = joblib.load(meta_path)
            expected_feature_names = model_meta.get('features_in_order', [])
            
            # Formdan gelen değerleri özellik sözlüğüne ekle
            for key in expected_feature_names:
                features_to_predict[key] = request.form.get(key)
            
            if not features_to_predict:
                flash('Model için beklenen özellikler yüklenemedi veya formdan özellik gelmedi.', 'danger')
                return render_template('test.html', model_types=model_types, current_model_type=model_type_selected, 
                                     form_features_dict=form_features_dict, trained_models=trained_models, submitted_features={})

            # Tüm beklenen özellikler mevcut mu kontrol et
            missing_features = [key for key in expected_feature_names if key not in features_to_predict or features_to_predict[key] is None]
            if missing_features:
                flash(f"Eksik özellikler: {', '.join(missing_features)}", 'danger')
                return render_template('test.html', model_types=model_types, current_model_type=model_type_selected,
                                     form_features_dict=form_features_dict, trained_models=trained_models, submitted_features=features_to_predict)

        except FileNotFoundError as e:
            flash(f"Model dosyası bulunamadı: {str(e)}", 'danger')
            print(traceback.format_exc())
            return render_template('test.html', model_types=model_types, current_model_type=model_type_selected, 
                                 form_features_dict=form_features_dict, trained_models=trained_models, submitted_features={})
        except Exception as e:
            flash(f"Özellikler alınırken bir hata oluştu: {str(e)}", 'danger')
            print(traceback.format_exc())
            return render_template('test.html', model_types=model_types, current_model_type=model_type_selected, 
                                 form_features_dict=form_features_dict, trained_models=trained_models, submitted_features={})

        if features_to_predict:
            try:
                # Eğitilmiş bir model seçildiyse onu kullan, yoksa genel model tipini kullan
                if trained_model_id:
                    # Önemli: burada model_id ile tahmin yapıyoruz, model_type değil
                    result = predict_phishing(features_to_predict, model_id=int(trained_model_id))
                    print(f"Kaydedilmiş model kullanılarak tahmin yapıldı. Model ID: {trained_model_id}")
                else:
                    result = predict_phishing(features_to_predict, model_type=model_type_selected)
                    print(f"Genel model tipi kullanılarak tahmin yapıldı. Model tipi: {model_type_selected}")
                
                # Tahmin sonrası form_features_dict'i tekrar yükleyelim ki doğru modelin özellikleri gösterilsin
                if trained_model_id:
                    trained_model = TrainedModel.query.get(trained_model_id)
                    model_path = trained_model.model_path
                    meta_path = model_path.replace("_model.joblib", "_meta.joblib")
                    feature_path = model_path.replace("_model.joblib", "_features.joblib")
                    
                    if os.path.exists(meta_path):
                        model_meta_for_form = joblib.load(meta_path)
                        all_features_from_meta = model_meta_for_form.get('features_in_order', [])
                    elif os.path.exists(feature_path):
                        all_features_from_meta = joblib.load(feature_path)
                    else:
                        all_features_from_meta = list(features_to_predict.keys())
                else:
                    meta_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_type_selected}_meta.joblib")
                    if os.path.exists(meta_path):
                        model_meta_for_form = joblib.load(meta_path)
                        all_features_from_meta = model_meta_for_form.get('features_in_order', [])
                    else:
                        all_features_from_meta = list(features_to_predict.keys())
                
                # Form için özellikleri güncelle
                    form_features_dict = {fn: {'label': fn.replace('_', ' ').title(), 'type': 'text', 'default': features_to_predict.get(fn, '')} for fn in all_features_from_meta}

                return render_template('test.html', result=result, model_types=model_types, current_model_type=model_type_selected, 
                                     form_features_dict=form_features_dict, trained_models=trained_models, submitted_features=features_to_predict)
            except Exception as e:
                flash(f'Tahmin sırasında bir hata oluştu: {str(e)}', 'danger')
                print(traceback.format_exc())
        else:
            flash('Lütfen test için özellikleri girin.', 'warning')
    
    # GET isteği için
    initial_model_type = request.args.get('model_type', model_types[0])
    form_features_dict_get = {}
    try:
        meta_path_get = os.path.join(app.config['MODEL_FOLDER'], f"{initial_model_type}_meta.joblib")
        if os.path.exists(meta_path_get):
            model_meta_get = joblib.load(meta_path_get)
            all_features_meta_get = model_meta_get.get('features_in_order', [])
            form_features_dict_get = {fn: {'label': fn.replace('_', ' ').title(), 'type': 'text', 'default': ''} for fn in all_features_meta_get}
        else:
             pass
    except Exception as e:
        print(f"Test formu (GET) için özellikler yüklenirken hata: {e}")

    return render_template('test.html', model_types=model_types, current_model_type=initial_model_type, 
                         form_features_dict=form_features_dict_get, trained_models=trained_models, submitted_features={})

@app.route('/api/model-metrics/<model_type>', methods=['GET'])
def get_model_metrics_route(model_type):
    meta_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_type}_meta.joblib")
    try:
        model_meta = joblib.load(meta_path)
        metrics = {
            'accuracy': model_meta.get('accuracy', 0),
            'precision': model_meta.get('precision', 0) * 100 if model_meta.get('precision') is not None else 0,
            'recall': model_meta.get('recall', 0) * 100 if model_meta.get('recall') is not None else 0,
            'f1': model_meta.get('f1', 0) * 100 if model_meta.get('f1') is not None else 0,
            'cv_mean': model_meta.get('cv_mean', 0),
            'cv_std': model_meta.get('cv_std', 0),
            'best_params': model_meta.get('best_params', {}),
            'column_types': {
                'numeric': len(model_meta.get('numeric_columns', [])),
                'categorical': len(model_meta.get('categorical_columns', [])),
                'text': len(model_meta.get('text_columns', []))
            },
            'features_in_order': model_meta.get('features_in_order', []),
            'target_classes': model_meta.get('target_classes', []),
            'class_report_str': model_meta.get('class_report_str', 'Rapor bulunamadı.')
        }
        return metrics
    except FileNotFoundError:
        return {"error": f"'{model_type.upper()}' modeli için meta dosyası bulunamadı."}, 404
    except Exception as e:
        print(f"API hata: Model bilgileri yüklenirken hata oluştu ({model_type}): {str(e)}")
        traceback.print_exc()
        return {"error": f"Model bilgileri yüklenirken bir sorun oluştu: {str(e)}"}, 500

@app.route('/api/trained-model-metrics/<int:model_id>', methods=['GET'])
def get_trained_model_metrics_route(model_id):
    try:
        # Veritabanından modeli bul
        trained_model = TrainedModel.query.get(model_id)
        if not trained_model:
            return {"error": f"ID: {model_id} olan model bulunamadı."}, 404
        
        # Model bilgilerini döndür
        metrics = {
            'accuracy': trained_model.accuracy,
            'precision': trained_model.precision,
            'recall': trained_model.recall,
            'f1': trained_model.f1_score,
            'train_count': trained_model.train_samples_count,
            'test_count': trained_model.test_samples_count,
            'trained_at': trained_model.trained_at.strftime("%Y-%m-%d %H:%M:%S") if trained_model.trained_at else "Bilinmiyor",
            'model_name': trained_model.model_name,
            'model_name_display': trained_model.model_name_display,
            'dataset_name': trained_model.dataset_name
        }
        
        return metrics
    except Exception as e:
        print(f"API hata: Eğitilmiş model bilgileri yüklenirken hata oluştu (ID: {model_id}): {str(e)}")
        traceback.print_exc()
        return {"error": f"Eğitilmiş model bilgileri yüklenirken bir sorun oluştu: {str(e)}"}, 500

if __name__ == '__main__':
    app.run(debug=True, port=5051) 