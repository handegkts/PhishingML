import os
import pandas as pd
import time
import joblib
from datetime import datetime
import pytz
from sqlalchemy.exc import IntegrityError
from flask import (
    Blueprint, flash, redirect, render_template, request, 
    url_for, current_app, send_from_directory, session, jsonify
)
from werkzeug.utils import secure_filename
from .models import train_model, predict_phishing, TrainedModel, db, get_model_icon_class, get_current_turkey_time
import traceback
import json

bp = Blueprint('phishing', __name__)

ALLOWED_EXTENSIONS = {'csv'}

# Tarih formatlama yardımcı fonksiyonu
def format_turkey_date(date_obj):
    """Tarih nesnesini Türkiye formatında formatlayarak string olarak döndürür."""
    if not date_obj:
        return ""
    if not date_obj.tzinfo:
        # Zaman dilimi bilgisi olmayan tarihlere Türkiye zaman dilimi ekle
        turkey_tz = pytz.timezone('Europe/Istanbul')
        date_obj = turkey_tz.localize(date_obj)
    return date_obj.strftime('%d.%m.%Y %H:%M')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # Eğer form eğitim sonuçlarını kaydetme adımı ise (save_model.html'den geliyorsa)
        if 'save_model' in request.form and request.form.get('save_model') == 'true':
            save_decision = request.form.get('save_decision', 'no')
            model_name_display_input = request.form.get('model_name', '').strip()

            # Session'dan son eğitim sonucunu al
            session_results = session.get('training_results')
            if not session_results or not isinstance(session_results, list) or len(session_results) == 0:
                flash('Kaydedilecek aktif bir eğitim sonucu bulunamadı. Lütfen tekrar model eğitin.', 'danger')
                return redirect(url_for('phishing.train'))
            
            last_result = session_results[0] # Listenin ilk (ve tek) elemanını al

            # Mevcut eğitilmiş modellerin isimlerini al (isim çakışması kontrolü için)
            trained_models_db = TrainedModel.query.order_by(TrainedModel.trained_at.desc()).all()
            trained_model_names_json = json.dumps([tm.model_name_display for tm in trained_models_db])
            
            if save_decision == 'yes':
                if not model_name_display_input:
                    flash('Eğitim adı boş olamaz!', 'danger')
                    return render_template('save_model.html', results=session_results, model_name_display_input=model_name_display_input, trained_models_json=trained_model_names_json)

                existing_model = TrainedModel.query.filter_by(model_name_display=model_name_display_input).first()
                if existing_model:
                    flash(f'HATA: "{model_name_display_input}" adında bir eğitim zaten mevcut! Lütfen farklı bir isim seçin.', 'danger')
                    return render_template('save_model.html', results=session_results, model_name_display_input=model_name_display_input, trained_models_json=trained_model_names_json)
                
                try:
                    new_trained_model = TrainedModel(
                        model_name=last_result['model_type'], 
                        model_name_display=model_name_display_input,
                        dataset_name=last_result['dataset_name'],
                        accuracy=last_result['accuracy'],
                        precision=last_result['precision'],
                        recall=last_result['recall'],
                        f1_score=last_result['f1_score'],
                        training_time=last_result['training_time'],
                        model_path=last_result['model_path'],
                        icon_class=last_result.get('icon_class'),
                        # Random seed değerini al
                        random_seed=last_result.get('model_params', {}).get('random_seed', 42),
                        # Eğitim ve test verilerinin sayılarını doğru şekilde kaydet
                        train_samples_count=last_result.get('train_samples_count'),
                        test_samples_count=last_result.get('test_samples_count'),
                        training_data_json=last_result.get('train_data_json'),
                        test_data_json=last_result.get('test_data_json'),
                        training_logs_json=json.dumps(last_result.get('training_logs', [])), # Loglar session'dan alınıp JSON olarak kaydediliyor
                        model_params_json=json.dumps(last_result.get('model_params', {})), # Model parametrelerini JSON olarak kaydet
                        dataset_hash=last_result.get('dataset_hash', '') # Veri seti hash değerini ekle
                    )
                    db.session.add(new_trained_model)
                    db.session.commit()
                    flash(f'"{model_name_display_input}" modeli başarıyla kaydedildi!', 'success')
                    session.pop('training_results', None) # Başarılı kayıt sonrası session temizlenir
                    session.pop('last_result_for_save', None) # app.py için kullanılan session değişkenini de temizle
                except IntegrityError:
                    db.session.rollback()
                    flash(f'VERİTABANI HATASI: "{model_name_display_input}" adında bir eğitim zaten mevcut (IntegrityError)! Lütfen farklı bir isim seçin.', 'danger')
                    return render_template('save_model.html', results=session_results, model_name_display_input=model_name_display_input, trained_models_json=trained_model_names_json)
                except Exception as e:
                    db.session.rollback()
                    flash(f'Model kaydedilirken bir hata oluştu: {str(e)}', 'danger')
                    current_app.logger.error(f"Model kaydetme hatası: {e}\\n{traceback.format_exc()}")
                    return render_template('save_model.html', results=session_results, model_name_display_input=model_name_display_input, trained_models_json=trained_model_names_json)
            
            elif save_decision == 'no':
                flash('Model kaydedilmedi.', 'info')
                session.pop('training_results', None) # Kayıttan vazgeçilince de session temizlenir
                session.pop('last_result_for_save', None) # app.py için kullanılan session değişkenini de temizle
            
            return redirect(url_for('phishing.train')) # Her durumda train sayfasına yönlendir
        
        # Burası dosya yükleme ve model eğitimi kısmı (save_model.html'den GELMİYORSA)
        if 'file' not in request.files:
            flash('Dosya bulunamadı', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('Dosya seçilmedi', 'danger')
            return redirect(request.url)
        
        model_type = request.form.get('model_type')
        if not model_type:
            flash('Lütfen bir model seçmelisiniz', 'warning')
            return redirect(request.url)
            
        # Model hiperparametrelerini topla (train_model fonksiyonuna göndermek için)
        model_params_to_pass = {}
        model_params_to_pass['train_test_split'] = request.form.get('train_test_split', '80')
        model_params_to_pass['random_seed'] = request.form.get('random_seed', '42')
        if model_type == 'random_forest':
            model_params_to_pass['rf_n_estimators'] = request.form.get('rf_n_estimators', '100')
            model_params_to_pass['rf_max_depth'] = request.form.get('rf_max_depth')
            model_params_to_pass['rf_min_samples_split'] = request.form.get('rf_min_samples_split', '2')
            model_params_to_pass['rf_criterion'] = request.form.get('rf_criterion', 'gini')
        elif model_type == 'svm':
            model_params_to_pass['svm_C'] = request.form.get('svm_C', '1.0')
            model_params_to_pass['svm_kernel'] = request.form.get('svm_kernel', 'rbf')
            model_params_to_pass['svm_gamma'] = request.form.get('svm_gamma', 'scale')
        elif model_type == 'decision_tree':
            model_params_to_pass['dt_max_depth'] = request.form.get('dt_max_depth')
            model_params_to_pass['dt_min_samples_split'] = request.form.get('dt_min_samples_split', '2')
            model_params_to_pass['dt_criterion'] = request.form.get('dt_criterion', 'gini')
        elif model_type == 'naive_bayes':
            model_params_to_pass['nb_var_smoothing'] = request.form.get('nb_var_smoothing', '1e-9')
        elif model_type == 'knn':
            model_params_to_pass['knn_n_neighbors'] = request.form.get('knn_n_neighbors', '5')
            model_params_to_pass['knn_weights'] = request.form.get('knn_weights', 'uniform')
            model_params_to_pass['knn_metric'] = request.form.get('knn_metric', 'minkowski')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # CSV dosyasını kontrol et
            try:
                # Dosyanın doğru formatta olduğunu kontrol et
                df_check = pd.read_csv(file_path)
                if df_check.empty:
                    flash('Yüklenen CSV dosyası boş!', 'danger')
                    return redirect(request.url)
                
                # Metin sütunları için uyarı ver
                text_columns = []
                for col in df_check.columns:
                    if df_check[col].dtype == 'object':
                        text_columns.append(col)
                
                if text_columns:
                    current_app.logger.warning(f"Metin türünde sütunlar tespit edildi: {text_columns}. Bu sütunlar model eğitimi sırasında kaldırılacak.")
                    flash(f"Dikkat: CSV dosyasında {len(text_columns)} adet metin türünde sütun tespit edildi. Bu sütunlar modelden otomatik olarak çıkarılacak.", 'warning')
            
            except Exception as e:
                flash(f'CSV dosyası yüklenirken hata oluştu: {str(e)}', 'danger')
                current_app.logger.error(f"CSV dosyası kontrol hatası: {traceback.format_exc()}")
                return redirect(request.url)
            
            # Results değişkenini başta tanımla
            results = []
            
            try:
                start_time = time.time()
                # train_model çağrısına model_params_to_pass eklendi
                metrics_dict, model_path = train_model(file_path, model_type, model_params_to_pass)
                training_time = time.time() - start_time
                
                # Debug bilgileri
                current_app.logger.debug(f"DEBUG: metrics_dict anahtarları: {metrics_dict.keys()}")
                current_app.logger.debug(f"DEBUG: Alınan loglar: {metrics_dict.get('training_logs')}")
                
                model_result = {
                    'model_type': model_type,
                    # 'model_name' yerine 'model_name_display' kullanalım (kullanıcı dostu isim için)
                    # Ancak bu isim kaydetme aşamasında kullanıcı tarafından belirleniyor.
                    # Şimdilik sadece model_type'ı baz alalım, save_model.html'de daha iyi bir isim oluşturulabilir.
                    'model_name_display': model_type.replace('_', ' ').title() + " Eğitimi", 
                    'dataset_name': filename,
                    'accuracy': metrics_dict['accuracy'],
                    'precision': metrics_dict['precision'],
                    'recall': metrics_dict['recall'],
                    'f1_score': metrics_dict['f1'],
                    'training_time': training_time,
                    'trained_at': get_current_turkey_time().strftime('%d.%m.%Y %H:%M:%S'),
                    'model_path': model_path,
                    'icon_class': get_model_icon_class(model_type),
                    # Eğitim ve test veri sayılarını doğru şekilde al
                    'train_samples_count': metrics_dict.get('train_samples_count', 0),
                    'test_samples_count': metrics_dict.get('test_samples_count', 0),
                    'train_data_json': metrics_dict.get('train_data_json', '[]'),
                    'test_data_json': metrics_dict.get('test_data_json', '[]'),
                    'training_logs': metrics_dict.get('training_logs', []), # Loglar eklendi
                    'model_params': {
                        **metrics_dict.get('model_params', {}),  # Önce mevcut model parametrelerini al
                        'random_seed': metrics_dict.get('random_seed', 42),  # Sonra random_seed değerini ekle
                        'train_test_split': model_params_to_pass.get('train_test_split', '80')  # Veri bölme oranını da ekle
                    },
                    'dataset_hash': metrics_dict.get('dataset_hash', '') # Veri seti hash değerini ekle
                }
                
                results = [model_result] # results listesi tek elemanlı olacak
                
            except Exception as e:
                flash(f'{model_type.upper()} model eğitimi sırasında hata oluştu: {str(e)}', 'danger')
                current_app.logger.error(f"Model eğitimi hatası ({model_type}): {traceback.format_exc()}")
                results = [] # Hata durumunda results boş liste olsun
            
            if results: # Sadece başarılı eğitim sonrası flash mesajı ve session kaydı
                flash(f'{model_type.replace('_', ' ').title()} modeli başarıyla eğitildi.', 'success')
                # Her iki session değişkenine de kaydet (app.py ve routes.py uyumluluğu için)
                session['training_results'] = results # routes.py için
                session['last_result_for_save'] = results[0] # app.py için
                
                # trained_models_json, save_model.html'e gönderilmek üzere hazırlanmalı
                trained_models_db = TrainedModel.query.order_by(TrainedModel.trained_at.desc()).all()
                trained_model_names = [tm.model_name_display for tm in trained_models_db]
                
                # Eğitim tarihi bilgisini daha okunabilir formatta ekle
                for result in results:
                    # Tarih string olarak kaydedilmiş, doğrudan kullanabiliriz
                    result['formatted_date'] = result['trained_at']
                
                return render_template('save_model.html', results=results, trained_models_json=json.dumps(trained_model_names))
        else:
            flash('Sadece CSV dosyaları kabul edilmektedir', 'warning')
    else:
        # GET isteği için (sayfa ilk yüklendiğinde veya model kaydetmeden dönüldüğünde)
        if 'save_model' not in request.form: 
            session.pop('training_results', None)
            session.pop('last_result_for_save', None)  # app.py için kullanılan session değişkenini de temizle
    
    trained_models = TrainedModel.query.order_by(TrainedModel.trained_at.desc()).all()
    # trained_models_json GET isteğinde de gönderilmeli ki save_model.html'e bir şekilde ulaşılırsa JS hata vermesin
    # Ancak normalde save_model.html GET ile doğrudan çağrılmamalı, /train üzerinden POST sonrası render edilmeli.
    # Bu yüzden bu satır aslında gereksiz olabilir veya save_model.html'in GET ile erişimini engellemek daha doğru olur.
    # Şimdilik bırakıyorum, çünkü save_model.html'in render edildiği yerde bu değişkene ihtiyaç var.
    trained_model_names_for_js = [tm.model_name_display for tm in trained_models]
    
    # Tarih formatlarını hazırla
    for model in trained_models:
        model.formatted_date = format_turkey_date(model.trained_at)
    
    return render_template('train.html', trained_models=trained_models, trained_models_json=json.dumps(trained_model_names_for_js))


@bp.route('/delete_history_entry/<int:entry_id>', methods=['POST'])
def delete_history_entry_route(entry_id):
    model = TrainedModel.query.get_or_404(entry_id)
    
    # Dosyaları silmek istiyorsanız (opsiyonel)
    # try:
    #     if model.model_path and os.path.exists(model.model_path):
    #         os.remove(model.model_path)
    # except Exception as e:
    #     flash(f'Model dosyası silinirken hata oluştu: {str(e)}', 'warning')
    
    db.session.delete(model)
    db.session.commit()
    
    flash('Eğitim kaydı silindi', 'success')
    return redirect(url_for('phishing.train'))

@bp.route('/delete_all_history', methods=['POST'])
def delete_all_history_route():
    models = TrainedModel.query.all()
    
    for model in models:
        db.session.delete(model)
    
    db.session.commit()
    
    flash('Tüm eğitim geçmişi silindi', 'success')
    return redirect(url_for('phishing.train'))

@bp.route('/edit_model/<int:model_id>', methods=['POST'])
def edit_model_route(model_id):
    model = TrainedModel.query.get_or_404(model_id)
    
    new_model_name = request.form.get('edit_model_name', '').strip()
    
    if not new_model_name:
        flash('Model adı boş olamaz!', 'danger')
        return redirect(url_for('phishing.train'))
        
    # Yeni ismin benzersiz olup olmadığını kontrol et (mevcut modelin adı dışında)
    existing_model = TrainedModel.query.filter(
        TrainedModel.model_name_display == new_model_name,
        TrainedModel.id != model_id
    ).first()
    
    if existing_model:
        flash(f'"{new_model_name}" adında bir model zaten mevcut! Lütfen farklı bir isim seçin.', 'danger')
        return redirect(url_for('phishing.train'))
    
    try:
        model.model_name_display = new_model_name
        db.session.commit()
        flash(f'Model adı başarıyla "{new_model_name}" olarak güncellendi.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Model adı güncellenirken bir hata oluştu: {str(e)}', 'danger')
        current_app.logger.error(f"Model adı güncelleme hatası: {str(e)}\\n{traceback.format_exc()}")
    
    return redirect(url_for('phishing.train'))

@bp.route('/get_model_logs/<int:model_id>', methods=['GET'])
def get_model_logs(model_id):
    """Modele ait log kayıtlarını API endpoint olarak sunar"""
    model = TrainedModel.query.get_or_404(model_id)
    
    try:
        # JSON formatında saklanmış logları parse et
        if model.training_logs_json:
            logs = json.loads(model.training_logs_json)
        else:
            logs = []
            
        return jsonify({
            'success': True,
            'model_id': model_id,
            'model_name': model.model_name_display,
            'logs': logs
        })
    except Exception as e:
        current_app.logger.error(f"Log verileri getirilirken hata: {str(e)}\\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/get_model_details/<int:model_id>', methods=['GET'])
def get_model_details(model_id):
    """Modele ait detaylı bilgileri API endpoint olarak sunar"""
    model = TrainedModel.query.get_or_404(model_id)
    
    try:
        # Model tipine göre parametre açıklamaları
        model_param_descriptions = {
            # Tüm modeller için ortak parametreler
            'common': {
                'random_seed': 'Model eğitimindeki rastgeleliği kontrol eden değer',
                'train_test_split': 'Veri setinin eğitim için ayrılan yüzdesi'
            },
            'random_forest': {
                'n_estimators': 'Toplulukta kullanılan ağaç sayısı',
                'max_depth': 'Ağaçların maksimum derinliği',
                'min_samples_split': 'Bir düğümü bölmek için gereken minimum örnek sayısı',
                'criterion': 'Bölme kalitesini ölçme yöntemi'
            },
            'svm': {
                'C': 'Düzenleme parametresi (küçük değerler = daha yumuşak sınır)',
                'kernel': 'Çekirdek fonksiyonu türü (rbf, linear, poly, sigmoid)',
                'gamma': 'Rbf, poly ve sigmoid çekirdek fonksiyonları için çekirdek katsayısı'
            },
            'decision_tree': {
                'max_depth': 'Ağacın maksimum derinliği',
                'min_samples_split': 'Bir düğümü bölmek için gereken minimum örnek sayısı',
                'criterion': 'Bölme kalitesini ölçme yöntemi'
            },
            'naive_bayes': {
                'var_smoothing': 'Varyansa eklenen stabilite sabiti'
            },
            'knn': {
                'n_neighbors': 'Sınıflandırma için kullanılacak komşu sayısı',
                'weights': 'Komşuların oy ağırlığı (uniform=eşit, distance=mesafeye göre)',
                'metric': 'Mesafe ölçüm yöntemi (minkowski, euclidean, manhattan)'
            }
        }
        
        # Model parametrelerinin form alanlarıyla eşleştirilmesi (ön ek dönüşümü)
        param_name_mapping = {
            'random_forest': {
                'n_estimators': 'rf_n_estimators',
                'max_depth': 'rf_max_depth',
                'min_samples_split': 'rf_min_samples_split',
                'criterion': 'rf_criterion'
            },
            'svm': {
                'C': 'svm_C',
                'kernel': 'svm_kernel',
                'gamma': 'svm_gamma'
            },
            'decision_tree': {
                'max_depth': 'dt_max_depth',
                'min_samples_split': 'dt_min_samples_split',
                'criterion': 'dt_criterion'
            },
            'naive_bayes': {
                'var_smoothing': 'nb_var_smoothing'
            },
            'knn': {
                'n_neighbors': 'knn_n_neighbors',
                'weights': 'knn_weights',
                'metric': 'knn_metric'
            }
        }
        
        # Parametre değerlerinin insan tarafından okunabilir karşılıkları
        param_value_labels = {
            'criterion': {
                'gini': 'Gini Endeksi',
                'entropy': 'Entropi'
            },
            'kernel': {
                'rbf': 'Radial Basis Function (RBF)',
                'linear': 'Lineer',
                'poly': 'Polinom',
                'sigmoid': 'Sigmoid'
            },
            'weights': {
                'uniform': 'Eşit ağırlık',
                'distance': 'Mesafeye göre ağırlıklı'
            },
            'metric': {
                'minkowski': 'Minkowski',
                'euclidean': 'Öklid',
                'manhattan': 'Manhattan'
            }
        }
        
        # Metrikler hakkında açıklamalar
        metric_descriptions = {
            'accuracy': 'Doğru tahmin edilen örneklerin tüm örneklere oranı',
            'precision': 'Phishing olarak tahmin edilenlerin gerçekten phishing olma oranı',
            'recall': 'Gerçek phishing sitelerinin tespit edilme oranı',
            'f1_score': 'Kesinlik ve duyarlılığın harmonik ortalaması',
            'training_time': 'Modelin eğitim süresi (saniye)'
        }
        
        # Veri sayıları hakkında bilgiler
        sample_info = {
            'train_samples_count': model.train_samples_count,
            'test_samples_count': model.test_samples_count,
            'train_test_ratio': f"{model.train_samples_count/(model.train_samples_count + model.test_samples_count)*100:.1f}% / {model.test_samples_count/(model.train_samples_count + model.test_samples_count)*100:.1f}%" if model.train_samples_count and model.test_samples_count else "Bilinmiyor"
        }
        
        # Veri seti bilgileri
        dataset_info = {
            'dataset_name': model.dataset_name,
            'dataset_hash': model.dataset_hash or "Bulunamadı",
            'dataset_size': model.train_samples_count + model.test_samples_count if model.train_samples_count and model.test_samples_count else None
        }
        
        # Performans metrikleri
        metrics = {
            'accuracy': model.accuracy,
            'precision': model.precision,
            'recall': model.recall,
            'f1_score': model.f1_score,
            'training_time': model.training_time
        }
        
        # Eğitim veri seti ve model ek bilgileri
        training_info = {
            'train_test_split': None,  # Aşağıda doldurulacak
            'dataset_size': model.train_samples_count + model.test_samples_count if model.train_samples_count and model.test_samples_count else None,
            'model_file_path': model.model_path
        }
        
        # Model bilgileri
        model_info = {
            'model_id': model.id,
            'model_name': model.model_name_display,
            'model_type': model.model_name,
            'model_type_display': model.model_name.replace('_', ' ').title(),
            'dataset_name': model.dataset_name,
            'trained_at': format_turkey_date(model.trained_at),
            'random_seed': model.random_seed,  # Random seed değerini ekle
            'metrics': metrics,
            'sample_info': sample_info,
            'training_info': training_info,
            'metric_descriptions': metric_descriptions,
            'param_descriptions': model_param_descriptions.get(model.model_name, {}),
            'param_name_mapping': param_name_mapping.get(model.model_name, {}),
            'param_value_labels': param_value_labels,
            'dataset_info': dataset_info
        }
        
        # Model parametrelerini JSON'dan parse et (eğer varsa)
        if model.model_params_json:
            try:
                model_params = json.loads(model.model_params_json)
                model_info['model_params'] = model_params
                
                # Veri bölme oranını model parametrelerinden de alalım (eğer mevcutsa)
                if 'train_test_split' in model_params:
                    model_info['training_info']['train_test_split'] = float(model_params.get('train_test_split', 80))
                
                # Model parametrelerini kullanıcı dostu gösterim için düzenle
                model_info['model_params_display'] = {}
                for param_name, param_value in model_params.items():
                    # Parametreleri düzenleme
                    display_value = param_value
                    
                    # Bazı özel parametreler için insan tarafından okunabilir değerler atayalım
                    if param_name in param_value_labels and param_value in param_value_labels[param_name]:
                        display_value = param_value_labels[param_name][param_value]
                    
                    model_info['model_params_display'][param_name] = {
                        'value': param_value,
                        'display_value': display_value,
                        'description': model_param_descriptions.get(model.model_name, {}).get(param_name, 'Parametre açıklaması bulunamadı')
                    }
            except Exception as e:
                current_app.logger.error(f"Model parametreleri parse edilirken hata: {str(e)}")
                model_info['model_params'] = {}
                model_info['model_params_display'] = {}
        else:
            model_info['model_params'] = {}
            model_info['model_params_display'] = {}
            
        return jsonify({
            'success': True,
            'model_info': model_info
        })
    except Exception as e:
        current_app.logger.error(f"Model detayları getirilirken hata: {str(e)}\\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
