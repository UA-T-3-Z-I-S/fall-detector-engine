import os
import sys
from dotenv import load_dotenv

# === BASE_DIR ===
if getattr(sys, 'frozen', False):  # Si se ejecuta desde el .exe
    BASE_DIR = sys._MEIPASS  # Carpeta temporal del .exe
    PROJECT_ROOT = BASE_DIR
else:  # Desarrollo
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/config/
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# === CONFIGURACIÓN DE MODELOS ===
MODEL_PATH_CNN = os.path.join(PROJECT_ROOT, "src", "keras", "model_cnn.keras")
MODEL_PATH_LSTM = os.path.join(PROJECT_ROOT, "src", "keras", "model_lstm.keras")
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "keras", "final_model.keras")

# Si es ejecutable, los modelos se buscan dentro del bundle (.exe)
if getattr(sys, 'frozen', False):
    MODEL_PATH_CNN = os.path.join(BASE_DIR, "keras", "model_cnn.keras")
    MODEL_PATH_LSTM = os.path.join(BASE_DIR, "keras", "model_lstm.keras")
    MODEL_PATH = os.path.join(BASE_DIR, "keras", "final_model.keras")

# === CONFIG LOCAL ===
if getattr(sys, 'frozen', False):
    # Busca el JSON en el mismo directorio donde esté el .exe
    CONFIG_PATH = os.path.join(os.getcwd(), "config_local.json")
else:
    # En desarrollo, sigue usando la raíz del proyecto
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "config_local.json")

