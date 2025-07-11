import numpy as np
import time  # ⏱️ Para medir tiempo
from keras.models import load_model
from config.paths import MODEL_PATH_CNN, MODEL_PATH_LSTM

class FallDetectorSeparated:
    def __init__(self, threshold=0.5):
        print("[📦] Cargando submodelos...")
        self.cnn = load_model(MODEL_PATH_CNN)
        self.lstm = load_model(MODEL_PATH_LSTM)
        self.threshold = threshold

    def predict_video(self, buffers):
        if not buffers:
            return {
                'caida': False,
                'buffers_totales': 0,
                'probabilidad_final': 0.0,
                'porcentaje': 0.0,
                'predicciones': [],
                'probabilidades': [],
                'tiempo_total': 0.0,
                'tiempo_cnn': 0.0,
                'tiempo_lstm': 0.0,
            }

        # [⏱️] Inicia cronómetro total
        tiempo_total_inicio = time.time()

        # [🚀] Extraer embeddings en lote con CNN
        x = np.array(buffers)  # (num_buffers, 16, 224, 224, 3)
        tiempo_cnn_inicio = time.time()
        embeddings_batch = self.cnn.predict(x, verbose=0)
        tiempo_cnn = time.time() - tiempo_cnn_inicio

        # [🔮] Predecir todas las secuencias en lote con LSTM
        tiempo_lstm_inicio = time.time()
        probs = self.lstm.predict(embeddings_batch, verbose=0)  # (num_buffers, 1)
        tiempo_lstm = time.time() - tiempo_lstm_inicio

        probs = np.atleast_1d(np.squeeze(probs))  # Asegura array 1D
        probabilidades = probs.tolist()
        predicciones = (probs >= self.threshold).astype(int).tolist()

        positivos = sum(predicciones)
        total = len(buffers)
        porcentaje = positivos / total if total > 0 else 0.0
        tiempo_total = time.time() - tiempo_total_inicio

        return {
            'caida': positivos > 0,
            'buffers_totales': total,
            'probabilidad_final': max(probabilidades),
            'porcentaje': porcentaje,
            'predicciones': predicciones,
            'probabilidades': probabilidades,
            'tiempo_total': tiempo_total,
            'tiempo_cnn': tiempo_cnn,
            'tiempo_lstm': tiempo_lstm,
        }
