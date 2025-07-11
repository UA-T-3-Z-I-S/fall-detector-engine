import cv2
import time
import os
from preprocessing.loader_opencv import load_video_frames
from preprocessing.buffer_creator import create_buffers
from utils.video_random import seleccionar_video_aleatorio
from utils.label import obtener_etiqueta_real

video_control = {'accion': None}

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _, h = param['canvas_size']
        if 20 <= x <= 100 and h - 40 <= y <= h:
            video_control['accion'] = 'repetir'
        elif 110 <= x <= 230 and h - 40 <= y <= h:
            video_control['accion'] = 'siguiente'
        elif 240 <= x <= 320 and h - 40 <= y <= h:
            video_control['accion'] = 'salir'

def mostrar_prediccion_en_video(frames, predicciones, probabilidades=None, video_path=None, caida_detectada=False, detector=None):
    assert detector is not None, "❌ Error: el modelo 'detector' no fue proporcionado."

    total_frames = len(frames)
    total_buffers = len(predicciones)
    buffer_size = total_frames // total_buffers if total_buffers > 0 else 1
    caida_count = sum(predicciones) if predicciones else 0
    video_name = os.path.basename(video_path) if video_path else "Video"

    canvas_width = 720
    video_height = 400
    panel_height = 180
    canvas_height = video_height + panel_height
    font = cv2.FONT_HERSHEY_SIMPLEX

    current_frame = 0
    paused = False

    cv2.namedWindow("Resultado")
    cv2.setMouseCallback("Resultado", mouse_event, param={'canvas_size': (canvas_width, canvas_height)})

    while True:
        if not paused and current_frame < total_frames:
            start_time = time.time()
            frame = frames[current_frame]
            buffer_idx = min(current_frame // buffer_size, total_buffers - 1) if total_buffers > 0 else 0
            label = predicciones[buffer_idx] if predicciones else int(caida_detectada)
            prob = probabilidades[buffer_idx] if probabilidades else None

            texto = "CAIDA" if label == 1 else "NO CAIDA"
            color = (0, 0, 255) if label == 1 else (0, 255, 0)

            frame_resized = cv2.resize(frame, (canvas_width, video_height))
            canvas = cv2.copyMakeBorder(frame_resized, 0, panel_height, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))

            cv2.putText(canvas, f"Video: {video_name}", (20, video_height + 30), font, 0.6, (255, 255, 255), 1)
            cv2.putText(canvas, f"Estado: {texto}", (20, video_height + 60), font, 0.7, color, 2)
            if prob is not None:
                cv2.putText(canvas, f"Probabilidad: {prob:.2f}", (250, video_height + 60), font, 0.6, color, 1)

            fps = 1.0 / (time.time() - start_time + 1e-8)
            cv2.putText(canvas, f"FPS: {fps:.2f}", (600, video_height + 30), font, 0.5, (200, 200, 200), 1)

            if predicciones:
                cv2.putText(canvas, f"Caidas detectadas: {caida_count}/{total_buffers}", (20, video_height + 90), font, 0.6, (0, 200, 255), 1)

            progreso = int((current_frame + 1) / total_frames * (canvas_width - 40))
            cv2.rectangle(canvas, (20, video_height + 110), (canvas_width - 20, video_height + 130), (50, 50, 50), -1)
            cv2.rectangle(canvas, (20, video_height + 110), (20 + progreso, video_height + 130), color, -1)

            # Botones
            cv2.rectangle(canvas, (20, canvas_height - 40), (100, canvas_height), (70, 70, 70), -1)
            cv2.putText(canvas, "Repetir", (30, canvas_height - 10), font, 0.6, (255, 255, 255), 1)

            cv2.rectangle(canvas, (110, canvas_height - 40), (230, canvas_height), (70, 70, 70), -1)
            cv2.putText(canvas, "Siguiente", (120, canvas_height - 10), font, 0.6, (255, 255, 255), 1)

            cv2.rectangle(canvas, (240, canvas_height - 40), (320, canvas_height), (70, 70, 70), -1)
            cv2.putText(canvas, "Salir", (250, canvas_height - 10), font, 0.6, (255, 255, 255), 1)

            cv2.imshow("Resultado", canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == 81:  # ←
            current_frame = max(0, current_frame - 5)
        elif key == 83:  # →
            current_frame = min(total_frames - 1, current_frame + 5)

        if not paused:
            current_frame += 1

        if video_control['accion']:
            action = video_control['accion']
            video_control['accion'] = None

            if action == 'repetir':
                current_frame = 0

            elif action == 'siguiente':
                cv2.destroyAllWindows()
                video_path = seleccionar_video_aleatorio()
                print(f"\n🎥 Video analizado: {video_path}")

                start_time = time.time()
                frames = load_video_frames(video_path)
                buffers = create_buffers(frames)
                result = detector.predict_video(buffers)
                end_time = time.time()

                print(f"[⏱] Tiempo total de procesamiento: {end_time - start_time:.2f} segundos")
                print(f"[✔] Voto final: {sum(result['predicciones'])} de {result['buffers_totales']} buffers indican caida "
                      f"(ratio: {result['porcentaje']:.2f})")
                print("[🔴] 🔔 ALARMA: CAIDA DETECTADA" if result['caida'] else "[🟢] No se detectó caida.")

                clase_real = obtener_etiqueta_real(video_path)
                if clase_real is not None:
                    print("[✅] Clasificación correcta ✅" if clase_real == int(result['caida']) else "[❌] Clasificación incorrecta ❌")

                mostrar_prediccion_en_video(
                    frames,
                    result['predicciones'],
                    result['probabilidades'],
                    video_path=video_path,
                    caida_detectada=result['caida'],
                    detector=detector  # 🔁 ¡SE PASA OTRA VEZ!
                )
                return

            elif action == 'salir':
                break

    cv2.destroyAllWindows()
