import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# =========================
# RUTAS DE VIDEO DE ENTRADA
# =========================

VIDEO_SOURCES = {
    'test': {
        'caida': os.getenv('DATASET_TEST_CAIDA'),
        'no_caida': os.getenv('DATASET_TEST_NO_CAIDA'),
    },
    'camera': os.getenv('LIVE_CAMERA_URL')  # Ruta RTSP/HTTP opcional
}

# ====================
# RUTA DEL MODELO IA
# ====================

MODEL_PATH = os.getenv(
    'MODEL_PATH',
    os.path.join('keras', 'fall_detector_model.keras')  # Fallback por defecto
)

# =============================
# SUBMODELOS OPTIMIZADOS IA
# =============================

MODEL_PATH_CNN = os.getenv(
    'MODEL_PATH_CNN',
    os.path.join('keras', 'modelo_cnn.keras')
)

MODEL_PATH_LSTM = os.getenv(
    'MODEL_PATH_LSTM',
    os.path.join('keras', 'modelo_lstm.keras')
)
