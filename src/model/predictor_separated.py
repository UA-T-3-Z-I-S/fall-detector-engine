import sys
import os
import numpy as np
import time
from datetime import datetime
from keras.models import load_model

# === Asegurarse de que 'src' esté en sys.path ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))  # sube un nivel desde model/
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Ahora podemos importar paths de forma absoluta
from config.paths import MODEL_PATH_CNN, MODEL_PATH_LSTM


class FallDetectorSeparated:
    def __init__(self, threshold=0.5):
        print("[Cargando submodelos CNN + LSTM]")
        self.cnn = load_model(MODEL_PATH_CNN)
        self.lstm = load_model(MODEL_PATH_LSTM)
        self.threshold = threshold

    def predict_video(self, buffers):
        timestamp_buffer = datetime.utcnow().isoformat() + "Z"

        if not buffers:
            return {
                "caida": False,
                "buffers_totales": 0,
                "probabilidad_final": 0.0,
                "porcentaje": 0.0,
                "timestamp_buffer": timestamp_buffer
            }

        tiempo_inicio_total = time.time()

        # --- Etapa CNN ---
        tiempo_inicio_cnn = time.time()
        x = np.array(buffers)
        embeddings_batch = self.cnn.predict(x, verbose=0)
        tiempo_cnn = time.time() - tiempo_inicio_cnn

        # --- Etapa LSTM ---
        tiempo_inicio_lstm = time.time()
        probs = self.lstm.predict(embeddings_batch, verbose=0)
        tiempo_lstm = time.time() - tiempo_inicio_lstm

        tiempo_total = time.time() - tiempo_inicio_total

        probs = np.atleast_1d(np.squeeze(probs))
        probabilidades = probs.tolist()
        predicciones = (probs >= self.threshold).astype(int).tolist()

        positivos = sum(predicciones)
        total = len(buffers)
        porcentaje = positivos / total if total > 0 else 0.0
        caida_detectada = positivos > 0

        resultado = {
            "caida": caida_detectada,
            "buffers_totales": total,
            "probabilidad_final": max(probabilidades),
            "porcentaje": porcentaje,
            "predicciones": predicciones,
            "probabilidades": probabilidades,
            "timestamp_buffer": timestamp_buffer
        }

        if caida_detectada:
            resultado.update({
                "tiempo_cnn": tiempo_cnn,
                "tiempo_lstm": tiempo_lstm,
                "tiempo_total": tiempo_total
            })

        return resultado
