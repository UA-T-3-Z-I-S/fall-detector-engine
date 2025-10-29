import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))  # ahora apunta a fall-detector-engine
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

MODEL_PATH_CNN = os.path.join(PROJECT_ROOT, os.getenv("MODEL_PATH_CNN", "src/keras/model_cnn.keras"))
MODEL_PATH_LSTM = os.path.join(PROJECT_ROOT, os.getenv("MODEL_PATH_LSTM", "src/keras/model_lstm.keras"))
MODEL_PATH = os.path.join(PROJECT_ROOT, os.getenv("MODEL_PATH", "src/keras/final_model.keras"))
