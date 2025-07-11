import numpy as np
from keras.models import load_model
from config.paths import MODEL_PATH_CNN, MODEL_PATH_LSTM

class FallDetectorSeparated:
    def __init__(self, threshold=0.5):
        print("[📦] Cargando submodelos...")
        self.cnn = load_model(MODEL_PATH_CNN)
        self.lstm = load_model(MODEL_PATH_LSTM)
        self.threshold = threshold

    def extract_embeddings(self, buffers):
        """
        Aplica CNN frame por frame en cada buffer y retorna embeddings.
        Entrada: (num_buffers, 16, 224, 224, 3)
        Salida:  (num_buffers, 16, embedding_size)
        """
        x = np.array(buffers)
        embeddings = self.cnn.predict(x, verbose=0)
        return embeddings

    def predict_sequence(self, embeddings):
        """
        Aplica el modelo LSTM sobre la secuencia completa de embeddings.
        Entrada: (16, embedding_size) o (1, 16, embedding_size)
        Salida: probabilidad de caída
        """
        embeddings = np.squeeze(embeddings)

        if embeddings.ndim == 2:
            x = np.expand_dims(embeddings, axis=0)  # (1, 16, embedding_size)
        elif embeddings.ndim == 3:
            x = embeddings  # (1, 16, embedding_size)
        else:
            raise ValueError(f"❌ Dimensión inesperada en embeddings: {embeddings.shape}")

        prob = self.lstm.predict(x, verbose=0)

        # ⚠️ Asegura que se extrae valor escalar
        if prob.ndim == 2 and prob.shape[1] == 1:
            prob = float(prob[0][0])
        elif prob.ndim == 1:
            prob = float(prob[0])
        else:
            raise ValueError(f"❌ Forma inesperada de salida en LSTM: {prob.shape}")

        return (1 if prob >= self.threshold else 0), prob

    def predict_video(self, buffers):
        if not buffers:
            return {
                'caida': False,
                'buffers_totales': 0,
                'probabilidad_final': 0.0,
                'porcentaje': 0.0,
                'predicciones': [],
                'probabilidades': []
            }

        predicciones = []
        probabilidades = []

        for buffer in buffers:
            embeddings = self.extract_embeddings([buffer])  # (1, 16, 224, 224, 3)
            label, prob = self.predict_sequence(embeddings)
            predicciones.append(label)
            probabilidades.append(prob)

        positivos = sum(predicciones)
        total = len(buffers)
        porcentaje = positivos / total if total > 0 else 0.0

        return {
            'caida': positivos > 0,
            'buffers_totales': total,
            'probabilidad_final': max(probabilidades),
            'porcentaje': porcentaje,
            'predicciones': predicciones,
            'probabilidades': probabilidades
        }
