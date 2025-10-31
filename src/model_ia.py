import os
import sys
import json
import cv2
import threading
import datetime
import time
import numpy as np
from dotenv import load_dotenv

# === AJUSTE DE RUTAS ===
BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from config.paths import MODEL_PATH_CNN, MODEL_PATH_LSTM, CONFIG_PATH
from preprocessing.buffer_creator import create_buffers
from model.predictor_separated import FallDetectorSeparated

# ==========================
# VARIABLES GLOBALES
# ==========================
detector = None
current_rtsp = ""
current_camera = "CAM-DESCONOCIDA"
running = True
lock = threading.Lock()
camera_ready = False
last_fall_time = 0
cooldown_seconds = 60
display_size = (640, 360)  # valor por defecto para vista de cámara

# Parámetros que se pueden configurar vía config_local.json
DEFAULTS = {
    "PREDICT_WORKERS": 2,
    "MIN_DETECTION_PERCENTAGE": 0.4,  # si porcentaje < 0.4 descartamos resultado
    "FRAME_QUEUE_MAX": 128,
    "BUFFER_QUEUE_MAX": 8,
    "FRAMES_PER_BATCH": 32,
    "BUFFER_SIZE": 16,
    "FRAME_TIMEOUT_SECONDS": 3.0,
    "RECONNECT_AFTER_INVALIDS": 15,
    "DISPLAY_SIZE": [640, 360],
    "COOLDOWN_SECONDS": 60
}

# ==========================
# FUNCIONES AUXILIARES
# ==========================
def emitir_evento(evento, data=None):
    """Envía salida JSON (para Electron o consola)"""
    payload = {"evento": evento}
    if data:
        payload.update(data)
    print(json.dumps(payload), flush=True)


def cargar_config_local():
    """Carga config_local.json con RTSP, cámara, cooldown y parámetros de pipeline"""
    global current_rtsp, current_camera, camera_ready, cooldown_seconds, display_size, DEFAULTS
    try:
        if not os.path.exists(CONFIG_PATH):
            emitir_evento("advertencia", {"mensaje": f"No se encontró {CONFIG_PATH}. Esperando configuración inicial..."})
            camera_ready = False
            return

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        current_rtsp = cfg.get("LIVE_CAMERA_URL", "")
        current_camera = cfg.get("CAMERA_NAME", "CAM-DESCONOCIDA")
        cooldown_seconds = cfg.get("COOLDOWN_SECONDS", DEFAULTS["COOLDOWN_SECONDS"])
        display_size = tuple(cfg.get("DISPLAY_SIZE", DEFAULTS["DISPLAY_SIZE"]))

        # Pipeline tunables
        DEFAULTS["PREDICT_WORKERS"] = int(cfg.get("PREDICT_WORKERS", DEFAULTS["PREDICT_WORKERS"]))
        DEFAULTS["MIN_DETECTION_PERCENTAGE"] = float(cfg.get("MIN_DETECTION_PERCENTAGE", DEFAULTS["MIN_DETECTION_PERCENTAGE"]))
        DEFAULTS["FRAME_QUEUE_MAX"] = int(cfg.get("FRAME_QUEUE_MAX", DEFAULTS["FRAME_QUEUE_MAX"]))
        DEFAULTS["BUFFER_QUEUE_MAX"] = int(cfg.get("BUFFER_QUEUE_MAX", DEFAULTS["BUFFER_QUEUE_MAX"]))
        DEFAULTS["FRAMES_PER_BATCH"] = int(cfg.get("FRAMES_PER_BATCH", DEFAULTS["FRAMES_PER_BATCH"]))
        DEFAULTS["BUFFER_SIZE"] = int(cfg.get("BUFFER_SIZE", DEFAULTS["BUFFER_SIZE"]))
        DEFAULTS["FRAME_TIMEOUT_SECONDS"] = float(cfg.get("FRAME_TIMEOUT_SECONDS", DEFAULTS["FRAME_TIMEOUT_SECONDS"]))
        DEFAULTS["RECONNECT_AFTER_INVALIDS"] = int(cfg.get("RECONNECT_AFTER_INVALIDS", DEFAULTS["RECONNECT_AFTER_INVALIDS"]))

        if not current_rtsp.strip():
            emitir_evento("advertencia", {"mensaje": "No se ha configurado una cámara RTSP. Esperando configuración..."})
            camera_ready = False
        else:
            camera_ready = True
            emitir_evento("config_cargada", {
                "camara": current_camera,
                "rtsp": current_rtsp,
                "cooldown": cooldown_seconds,
                "display_size": display_size,
                "predict_workers": DEFAULTS["PREDICT_WORKERS"],
                "min_detection_percentage": DEFAULTS["MIN_DETECTION_PERCENTAGE"]
            })

    except Exception as e:
        emitir_evento("error", {"mensaje": f"No se pudo cargar config_local.json: {e}"})
        camera_ready = False


def inicializar_modelo():
    """Carga el modelo CNN/LSTM"""
    global detector
    try:
        detector = FallDetectorSeparated()
        emitir_evento("modelo_listo", {"modo": "separado"})
    except Exception as e:
        emitir_evento("error", {"mensaje": f"Error cargando modelo: {e}"})


def leer_frame_seguro_from_cap(cap):
    """Lee un frame y valida que no esté corrupto (versión directa usada por reader)"""
    try:
        ret, frame = cap.read()
    except Exception:
        return None
    if not ret or frame is None:
        return None
    if getattr(frame, "size", 0) == 0:
        return None
    h, w = frame.shape[:2]
    if h < 100 or w < 100:
        return None
    return frame


# ==========================
# DETECCIÓN EN TIEMPO REAL (Pipeline multihilo)
# ==========================
def detectar_caidas():
    import queue

    global running, current_rtsp, current_camera, camera_ready, last_fall_time

    # Cargar parámetros locales desde DEFAULTS (pueden actualizarse con reload_config)
    FRAME_QUEUE_MAX = DEFAULTS["FRAME_QUEUE_MAX"]
    BUFFER_QUEUE_MAX = DEFAULTS["BUFFER_QUEUE_MAX"]
    FRAMES_PER_BATCH = DEFAULTS["FRAMES_PER_BATCH"]
    BUFFER_SIZE = DEFAULTS["BUFFER_SIZE"]
    NUM_PREDICT_WORKERS = DEFAULTS["PREDICT_WORKERS"]
    FRAME_TIMEOUT_SECONDS = DEFAULTS["FRAME_TIMEOUT_SECONDS"]
    RECONNECT_AFTER_INVALIDS = DEFAULTS["RECONNECT_AFTER_INVALIDS"]
    MIN_DETECTION_PERCENTAGE = DEFAULTS["MIN_DETECTION_PERCENTAGE"]

    # Colas de la pipeline
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX)
    buffer_queue = queue.Queue(maxsize=BUFFER_QUEUE_MAX)

    stop_threads = threading.Event()

    # Telemetría simple
    telemetry = {
        "frames_read": 0,
        "frames_dropped_queue_full": 0,
        "buffers_produced": 0,
        "buffers_dropped_full": 0,
        "buffers_processed": 0
    }

    def start_reader(rtsp_url):
        """Hilo que lee frames de la cámara y los pone en frame_queue (drop si full)."""
        nonlocal telemetry
        cap = None
        try:
            url = rtsp_url
            # Usa stream2 si aparece stream1 para intentar baja latencia
            if "stream1" in url:
                url = url.replace("stream1", "stream2")

            # Opciones FFmpeg para baja latencia
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|buffer_size;1024|max_delay;500000|stimeout;3000000|fflags;nobuffer"
            )

            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                emitir_evento("advertencia", {"mensaje": f"No se pudo abrir RTSP (reader): {url}"})
                return

            # lectura continua
            while not stop_threads.is_set() and running:
                frame = leer_frame_seguro_from_cap(cap)
                if frame is None:
                    time.sleep(0.005)
                    continue

                telemetry["frames_read"] += 1
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    telemetry["frames_dropped_queue_full"] += 1
                    # si la cola está llena, dropeamos el frame (evitar backlog)
                # pequeña pausa para dar margen a CPU
                time.sleep(0.001)

        except Exception as e:
            emitir_evento("error", {"mensaje": f"Reader thread error: {e}"})
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    def bufferizer():
        """Consume frame_queue, arma ventanas deslizantes de FRAMES_PER_BATCH y encola buffers listos."""
        nonlocal telemetry
        local_window = []
        while not stop_threads.is_set() and running:
            try:
                frame = frame_queue.get(timeout=1.0)
            except Exception:
                # timeout esperando frames, sigue loop
                continue

            local_window.append(frame)
            # mantener ventana deslizante
            if len(local_window) > FRAMES_PER_BATCH:
                local_window.pop(0)

            if len(local_window) >= FRAMES_PER_BATCH:
                try:
                    buffers = create_buffers(local_window, buffer_size=BUFFER_SIZE)
                except Exception as e:
                    emitir_evento("error", {"mensaje": f"bufferizer: create_buffers fallo: {e}"})
                    continue

                if not buffers:
                    continue

                telemetry["buffers_produced"] += 1
                try:
                    buffer_queue.put_nowait(buffers)
                except queue.Full:
                    telemetry["buffers_dropped_full"] += 1
                    # drop buffers para evitar acumulación de latencia

    def predictor_worker(worker_id):
        """Consume buffer_queue y ejecuta predict_video en paralelo."""
        nonlocal telemetry
        while not stop_threads.is_set() and running:
            try:
                buffers = buffer_queue.get(timeout=1.0)
            except Exception:
                continue

            telemetry["buffers_processed"] += 1
            try:
                result = detector.predict_video(buffers)
            except Exception as e:
                emitir_evento("error", {"mensaje": f"Worker {worker_id} predict fallo: {e}"})
                continue

            # Si porcentaje es bajo, descartamos el resultado (evita falsas alarmas y gasto)
            porcentaje = float(result.get("porcentaje", 0.0))
            # Si el predictor devuelve "buffers_totales" > 0 y porcentaje >= umbral procedemos
            if porcentaje >= MIN_DETECTION_PERCENTAGE:
                now = time.time()
                with lock:
                    if now - last_fall_time >= cooldown_seconds:
                        last_fall_time = now
                        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
                        emitir_evento("caida_detectada", {
                            "camara": current_camera,
                            "timestamp": timestamp,
                            "porcentaje": porcentaje,
                            "buffers_totales": result.get("buffers_totales", 0)
                        })
                    else:
                        emitir_evento("cooldown_activo", {
                            "camara": current_camera,
                            "restante": round(cooldown_seconds - (now - last_fall_time), 1)
                        })
            else:
                # descartado por baja confianza — opcional emitir telemetría mínima
                emitir_evento("resultado_descartado", {
                    "camara": current_camera,
                    "porcentaje": porcentaje,
                    "buffers_totales": result.get("buffers_totales", 0)
                })

    # ------------------------------
    # Bucle que administra la conexión y arranque de threads
    # ------------------------------
    while running:
        # actualizar parámetros por si hubo reload_config
        FRAME_QUEUE_MAX = DEFAULTS["FRAME_QUEUE_MAX"]
        BUFFER_QUEUE_MAX = DEFAULTS["BUFFER_QUEUE_MAX"]
        FRAMES_PER_BATCH = DEFAULTS["FRAMES_PER_BATCH"]
        BUFFER_SIZE = DEFAULTS["BUFFER_SIZE"]
        NUM_PREDICT_WORKERS = DEFAULTS["PREDICT_WORKERS"]
        FRAME_TIMEOUT_SECONDS = DEFAULTS["FRAME_TIMEOUT_SECONDS"]
        RECONNECT_AFTER_INVALIDS = DEFAULTS["RECONNECT_AFTER_INVALIDS"]
        MIN_DETECTION_PERCENTAGE = DEFAULTS["MIN_DETECTION_PERCENTAGE"]

        if not camera_ready or not current_rtsp:
            time.sleep(2)
            continue

        # vaciar colas previas
        try:
            while not frame_queue.empty():
                frame_queue.get_nowait()
        except Exception:
            pass
        try:
            while not buffer_queue.empty():
                buffer_queue.get_nowait()
        except Exception:
            pass

        stop_threads.clear()

        # start reader thread
        reader = threading.Thread(target=start_reader, args=(current_rtsp,), daemon=True)
        reader.start()

        # start bufferizer thread
        buf_thread = threading.Thread(target=bufferizer, daemon=True)
        buf_thread.start()

        # start predictor workers
        workers = []
        for i in range(NUM_PREDICT_WORKERS):
            w = threading.Thread(target=predictor_worker, args=(i,), daemon=True)
            w.start()
            workers.append(w)

        emitir_evento("camara_activa", {"camara": current_camera, "rtsp": current_rtsp})

        # supervisión: si no llegan frames válidos por un periodo, reiniciamos la conexión
        last_valid_frame_time = time.time()
        consecutive_failures = 0
        try:
            while running and camera_ready:
                # revisar si hay frames nuevos en la cola
                try:
                    # peek último frame sin quitarlo
                    with frame_queue.mutex:
                        last = frame_queue.queue[-1] if len(frame_queue.queue) > 0 else None
                except Exception:
                    last = None

                if last is not None:
                    last_valid_frame_time = time.time()

                if time.time() - last_valid_frame_time > FRAME_TIMEOUT_SECONDS:
                    consecutive_failures += 1
                    if consecutive_failures >= RECONNECT_AFTER_INVALIDS:
                        emitir_evento("advertencia", {"mensaje": f"Sin frames válidos ({consecutive_failures}). Reiniciando conexión..."})
                        break
                else:
                    consecutive_failures = 0

                # VISUALIZACION: si estamos en desarrollo mostramos la última frame (no la removemos)
                if not getattr(sys, 'frozen', False):
                    try:
                        with frame_queue.mutex:
                            if len(frame_queue.queue) > 0:
                                last_frame = frame_queue.queue[-1]
                                resized = cv2.resize(last_frame, display_size)
                                cv2.imshow(f"Vista de cámara ({current_camera})", resized)
                                cv2.resizeWindow(f"Vista de cámara ({current_camera})", *display_size)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    running = False
                                    break
                    except Exception:
                        pass

                # cada minuto imprimimos telemetría resumida (no candado)
                # (puedes desactivar o ajustar la frecuencia)
                now = time.time()
                if int(now) % 60 == 0:
                    # imprimir una sola vez por segundo con small guard
                    emitir_evento("telemetria", {
                        "frames_read": telemetry["frames_read"],
                        "frames_dropped_queue_full": telemetry["frames_dropped_queue_full"],
                        "buffers_produced": telemetry["buffers_produced"],
                        "buffers_dropped_full": telemetry["buffers_dropped_full"],
                        "buffers_processed": telemetry["buffers_processed"]
                    })
                    # reset ligero para no spamear
                    telemetry["frames_read"] = 0
                    telemetry["frames_dropped_queue_full"] = 0
                    telemetry["buffers_produced"] = 0
                    telemetry["buffers_dropped_full"] = 0
                    telemetry["buffers_processed"] = 0

                time.sleep(0.05)

        finally:
            # señalamos a threads que se detengan y esperamos su finalización corta
            stop_threads.set()
            try:
                reader.join(timeout=1.0)
            except Exception:
                pass
            try:
                buf_thread.join(timeout=1.0)
            except Exception:
                pass
            for w in workers:
                try:
                    w.join(timeout=1.0)
                except Exception:
                    pass

            # limpiar ventanas dev
            if not getattr(sys, 'frozen', False):
                cv2.destroyAllWindows()

            time.sleep(2)

    # Limpieza final
    stop_threads.set()
    if not getattr(sys, 'frozen', False):
        cv2.destroyAllWindows()


# ==========================
# COMANDOS DESDE ELECTRON
# ==========================
def procesar_comando(linea):
    global running, current_rtsp, current_camera, camera_ready
    try:
        data = json.loads(linea)
        comando = data.get("comando")

        if comando == "iniciar_modelo":
            current_rtsp = data.get("rtsp", current_rtsp)
            current_camera = data.get("camara", current_camera)
            camera_ready = bool(current_rtsp.strip())
            emitir_evento("modelo_iniciado", {"camara": current_camera, "rtsp": current_rtsp})

        elif comando == "cambiar_camara":
            current_rtsp = data.get("rtsp", current_rtsp)
            current_camera = data.get("camara", current_camera)
            camera_ready = bool(current_rtsp.strip())
            emitir_evento("camara_actualizada", {"camara": current_camera, "rtsp": current_rtsp})

        elif comando == "reload_config":
            cargar_config_local()

        elif comando == "detener_modelo":
            running = False
            emitir_evento("modelo_detenido")
            sys.exit(0)

    except Exception as e:
        emitir_evento("error", {"mensaje": f"Comando inválido: {e}"})


def loop_escucha():
    for linea in sys.stdin:
        procesar_comando(linea.strip())


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    emitir_evento("modelo_iniciando")
    cargar_config_local()
    inicializar_modelo()

    threading.Thread(target=detectar_caidas, daemon=True).start()
    loop_escucha()
