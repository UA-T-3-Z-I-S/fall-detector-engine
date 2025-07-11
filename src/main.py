# src/main.py

from model.predictor_separated import FallDetectorSeparated as FallDetector
from preprocessing.loader_opencv import load_video_frames
from preprocessing.buffer_creator import create_buffers
from utils.video_utils import verificar_video_mp4
from utils.label import obtener_etiqueta_real
from utils.video_random import seleccionar_video_aleatorio
from ui.visualizer import mostrar_prediccion_en_video
import time

# ✅ Cargamos el detector SOLO una vez
detector = FallDetector()

def main():
    while True:
        # 1. Seleccionar y validar video
        path = seleccionar_video_aleatorio()
        path = verificar_video_mp4(path)

        # 2. Cargar video y crear buffers
        frames = load_video_frames(path)
        buffers = create_buffers(frames)

        # 3. Inferir y medir tiempo
        start_pred = time.time()
        result = detector.predict_video(buffers)
        end_pred = time.time()

        # 4. Mostrar resultados en consola
        print(f"\n🎥 Video analizado: {path}")
        print(f"[⏱] Tiempo de predicción: {end_pred - start_pred:.2f} s")
        print(f"[✔] Buffers procesados: {result['buffers_totales']}")
        print(f"[📈] Probabilidad final: {result['probabilidad_final']:.4f}")
        print("[🔴] 🔔 ALARMA: CAÍDA DETECTADA" if result['caida'] else "[🟢] No se detectó caída.")

        clase_real = obtener_etiqueta_real(path)
        if clase_real is not None:
            if clase_real == int(result['caida']):
                print("[✅] Clasificación correcta ✅")
            else:
                print("[❌] Clasificación incorrecta ❌")
        else:
            print("[⚠️] No se pudo determinar la clase real desde la ruta del video.")

        # 5. Lanzar visualizador y pasar detector
        mostrar_prediccion_en_video(
            frames,
            result.get('predicciones', []),
            result.get('probabilidades', []),
            video_path=path,
            caida_detectada=result['caida'],
            detector=detector
        )

        break  # salir después de un ciclo

if __name__ == "__main__":
    main()
