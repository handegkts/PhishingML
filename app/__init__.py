import os
from flask import Flask, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_session import Session

# Veritabanı nesnesini burada oluştur
db = SQLAlchemy()
migrate = Migrate()
sess = Session()

def create_app(test_config=None):
    # Flask uygulamasını oluştur ve yapılandır
    app = Flask(__name__, instance_relative_config=True)
    
    # Varsayılan konfigürasyonu ayarla
    app.config.from_mapping(
        SECRET_KEY='gelistirme-anahtari',
        UPLOAD_FOLDER=os.path.join(app.instance_path, 'uploads'),
        MODEL_FOLDER=os.path.join(app.instance_path, 'models'),
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{os.path.join(app.instance_path, 'phishing.db')}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SESSION_TYPE='filesystem',
        PERMANENT_SESSION_LIFETIME=1800,  # 30 dakika session süresi
    )

    if test_config is None:
        # Eğer test yapılandırması yoksa, instance config'den yükle (varsa)
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Test için verilen yapılandırmayı yükle
        app.config.from_mapping(test_config)

    # Instance klasörünün var olduğundan emin ol
    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

    # Veritabanını başlat
    db.init_app(app)
    
    # Session'ı başlat
    sess.init_app(app)
    
    from . import models
    migrate.init_app(app, db)
    
    with app.app_context():
        # Otomatik şema oluşturma yerine artık migrate kullanacağız
        # Ancak ilk çalıştırmada veya test sırasında şema oluşturmak gerekebilir
        try:
            db.create_all()
        except:
            pass

    # Blueprint'leri kaydet
    from . import routes
    app.register_blueprint(routes.bp)

    return app
