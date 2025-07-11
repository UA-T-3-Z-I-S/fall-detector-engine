# main.py
from model.predictor import FallDetector
from preprocessing.loader_opencv import load_video_frames
from preprocessing.buffer_creator import create_buffers
from utils.video_utils import verificar_video_mp4
from utils.label import obtener_etiqueta_real
from utils.video_random import seleccionar_video_aleatorio
from ui.visualizer import mostrar_prediccion_en_video
import time

# ✅ Cargar el modelo SOLO una vez (¡fuera del bucle!)
detector = FallDetector()

def main():
    while True:
        # 1. Seleccionar video aleatorio
        path = seleccionar_video_aleatorio()
        path = verificar_video_mp4(path)

        # 2. Cargar video y procesar
        frames = load_video_frames(path)
        buffers = create_buffers(frames)

        # 🕒 Medir solo tiempo de predicción
        start_pred = time.time()
        result = detector.predict_video(buffers)
        end_pred = time.time()

        # 3. Mostrar resultados
        print(f"\n🎥 Video analizado: {path}")
        print(f"[⏱] Tiempo de predicción: {end_pred - start_pred:.2f} s")
        print(f"[✔] Voto final: {result['positivos']} de {result['buffers_totales']} buffers indican caída "
              f"(ratio: {result['porcentaje']:.2f})")
        print("[🔴] 🔔 ALARMA: CAÍDA DETECTADA" if result['caida'] else "[🟢] No se detectó caída.")

        clase_real = obtener_etiqueta_real(path)
        if clase_real is not None:
            if clase_real == int(result['caida']):
                print("[✅] Clasificación correcta ✅")
            else:
                print("[❌] Clasificación incorrecta ❌")
        else:
            print("[⚠️] No se pudo determinar la clase real desde la ruta del video.")

        # 4. Visualización con navegación (repetir / siguiente)
        mostrar_prediccion_en_video(
            frames,
            result['predicciones'],
            result['probabilidades'],
            video_path=path,
            caida_detectada=result['caida']
        )

        break  # salir después de un ciclo

if __name__ == "__main__":
    main()
