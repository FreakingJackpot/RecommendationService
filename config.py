import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SECRET_KEY = os.environ.get('SECRET_KEY', 'this is a secret')

# database settings
DATABASES = {
    'portal': {
        'name': os.environ.get('DB_NAME', 'postgres'),
        'user': os.environ.get('DB_USERNAME', 'postgres'),
        'password': os.environ.get('DB_PASSWORD', 'postgres'),
        'host': os.environ.get('DB_HOST', '127.0.0.1'),
        'life_time': os.environ.get('DB_CONN_LIFE_TIME', 3600),
    },
    'service':
        {
            'name': os.environ.get('DB_NAME', 'postgres'),
            'user': os.environ.get('DB_USERNAME', 'postgres'),
            'password': os.environ.get('DB_PASSWORD', 'postgres'),
            'host': os.environ.get('DB_HOST', '127.0.0.1'),
            'life_time': os.environ.get('DB_CONN_LIFE_TIME', 3600),
        }
}

# celery settings
CELERY = dict(
    broker_url="redis://localhost:6379",
    result_backend="redis://localhost:6379",
    timezone="UTC",
    task_ignore_result=True,
    beat_schedule={
        "time_scheduler": {
            "task": "app.train_model",
            # Run every second
            "schedule": 3600,
        }
    }
)

# recommender settings
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'training_data', 'checkpoints')
MODEL_DIR = os.path.join(BASE_DIR, 'training_data', 'outputs', 'model')
ENCODER_PATH = os.path.join(BASE_DIR, 'training_data', 'outputs', 'encoder.pkl')
