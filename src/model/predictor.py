import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf
tf.config.optimizer.set_jit(True)

import numpy as np
from keras.models import load_model
from config.paths import MODEL_PATH

class FallDetector:
    def __init__(self, threshold=0.5, vote_ratio=0.6):
        print("[📦] Cargando modelo desde:", MODEL_PATH)
        self.model = load_model(MODEL_PATH)
        self.model.trainable = False  # Solo inferencia
        self.threshold = threshold
        self.vote_ratio = vote_ratio

    def predict(self, buffer):
        x = np.expand_dims(buffer, axis=0)  # (1, 16, 224, 224, 3)
        prob = float(self.model.predict(x, verbose=0).squeeze())
        return 1 if prob >= self.threshold else 0, prob

    def predict_video(self, buffers):
        predicciones = []
        probabilidades = []

        for buffer in buffers:
            label, prob = self.predict(buffer)
            predicciones.append(label)
            probabilidades.append(prob)

        total = len(predicciones)
        positivos = sum(predicciones)
        ratio = positivos / total if total > 0 else 0

        return {
            'caida': ratio >= self.vote_ratio,
            'buffers_totales': total,
            'positivos': positivos,
            'porcentaje': ratio,
            'predicciones': predicciones,
            'probabilidades': probabilidades
        }
