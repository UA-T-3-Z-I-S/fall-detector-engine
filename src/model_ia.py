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
cooldown_until = 0  # <-- nuevo: marca hasta cuándo está activo el cooldown
display_size = (640, 360)  # valor por defecto

# Parámetros configurables desde config_local.json
DEFAULTS = {
    "PREDICT_WORKERS": 2,
    "MIN_DETECTION_PERCENTAGE": 0.6,  # relajado temporalmente: 60%
    "FRAME_QUEUE_MAX": 128,
    "BUFFER_QUEUE_MAX": 8,
    "FRAMES_PER_BATCH": 32,
    "BUFFER_SIZE": 16,
    "FRAME_TIMEOUT_SECONDS": 3.0,
    "RECONNECT_AFTER_INVALIDS": 15,
    "DISPLAY_SIZE": [640, 360],
    "COOLDOWN_SECONDS": 60,
    "MIN_MOTION": 0.02,   # menos estricto: frames normalizados [0,1]
    "MIN_CONSECUTIVE_DETECTIONS": 1  # permitir 1 detección consecutiva para pruebas
}

consecutive_detections = 0  # contador de detecciones consecutivas

# ==========================
# FUNCIONES AUXILIARES
# ==========================
def emitir_evento(evento, data=None):
    payload = {"evento": evento}
    if data:
        payload.update(data)
    print(json.dumps(payload), flush=True)

def cargar_config_local():
    global current_rtsp, current_camera, camera_ready, cooldown_seconds, display_size, DEFAULTS
    
    if not os.path.exists(CONFIG_PATH):
        config_inicial = {
            "LIVE_CAMERA_URL": "",
            "CAMERA_NAME": "CAM-DESCONOCIDA",
            "COOLDOWN_SECONDS": 60,
            "DISPLAY_SIZE": [640, 360],
            "PREDICT_WORKERS": 2,
            "MIN_DETECTION_PERCENTAGE": 0.4
        }
        try:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config_inicial, f, indent=2)
            emitir_evento("config_creada", {"path": CONFIG_PATH})
        except Exception as e:
            emitir_evento("error", {"mensaje": f"Error creando config: {str(e)}"})
            return False
            
    try:
        with open(CONFIG_PATH, 'r') as f:
            cfg = json.load(f)
            
        current_rtsp = cfg.get("LIVE_CAMERA_URL", "")
        current_camera = cfg.get("CAMERA_NAME", "CAM-DESCONOCIDA")
        cooldown_seconds = int(cfg.get("COOLDOWN_SECONDS", 60))
        display_size = tuple(cfg.get("DISPLAY_SIZE", [640, 360]))
        DEFAULTS["PREDICT_WORKERS"] = int(cfg.get("PREDICT_WORKERS", 2))
        DEFAULTS["MIN_DETECTION_PERCENTAGE"] = float(cfg.get("MIN_DETECTION_PERCENTAGE", 0.4))
        
        return True
        
    except Exception as e:
        emitir_evento("error", {"mensaje": f"Error cargando config: {str(e)}"})
        return False

def inicializar_modelo():
    global detector
    try:
        detector = FallDetectorSeparated()
        emitir_evento("modelo_listo", {"modo": "separado"})
    except Exception as e:
        emitir_evento("error", {"mensaje": f"Error cargando modelo: {e}"})

def leer_frame_seguro_from_cap(cap):
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
# DETECCIÓN EN TIEMPO REAL
# ==========================
def detectar_caidas():
    import queue
    global running, current_rtsp, current_camera, camera_ready, last_fall_time

    FRAME_QUEUE_MAX = DEFAULTS["FRAME_QUEUE_MAX"]
    BUFFER_QUEUE_MAX = DEFAULTS["BUFFER_QUEUE_MAX"]
    FRAMES_PER_BATCH = DEFAULTS["FRAMES_PER_BATCH"]
    BUFFER_SIZE = DEFAULTS["BUFFER_SIZE"]
    NUM_PREDICT_WORKERS = DEFAULTS["PREDICT_WORKERS"]
    FRAME_TIMEOUT_SECONDS = DEFAULTS["FRAME_TIMEOUT_SECONDS"]
    RECONNECT_AFTER_INVALIDS = DEFAULTS["RECONNECT_AFTER_INVALIDS"]
    MIN_DETECTION_PERCENTAGE = DEFAULTS["MIN_DETECTION_PERCENTAGE"]

    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX)
    buffer_queue = queue.Queue(maxsize=BUFFER_QUEUE_MAX)
    stop_threads = threading.Event()

    telemetry = {
        "frames_read": 0,
        "frames_dropped_queue_full": 0,
        "buffers_produced": 0,
        "buffers_dropped_full": 0,
        "buffers_processed": 0
    }

    def start_reader(rtsp_url):
        nonlocal telemetry
        cap = None
        local_ready = False

        try:
            url = rtsp_url
            if "stream1" in url:
                url = url.replace("stream1", "stream2")

            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|buffer_size;1024|max_delay;500000|stimeout;3000000|fflags;nobuffer"
            )

            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                emitir_evento("advertencia", {"mensaje": f"No se pudo abrir RTSP: {url}"})
                return

            # Esperar un frame válido
            start_time = time.time()
            while not local_ready and time.time() - start_time < 5:
                frame = leer_frame_seguro_from_cap(cap)
                if frame is not None:
                    local_ready = True
                    break
                time.sleep(0.05)

            if not local_ready:
                emitir_evento("advertencia", {"mensaje": "No se recibió ningún frame válido de la cámara"})
                return

            global camera_ready
            camera_ready = True
            emitir_evento("camara_activa", {"camara": current_camera, "rtsp": current_rtsp})

            # Lectura continua
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
                time.sleep(0.001)

        except Exception as e:
            emitir_evento("error", {"mensaje": f"Reader thread error: {e}"})
        finally:
            if cap is not None:
                cap.release()
            camera_ready = False

    # Bufferizer
    def bufferizer():
        nonlocal telemetry
        local_window = []
        while not stop_threads.is_set() and running:
            try:
                frame = frame_queue.get(timeout=1.0)
            except Exception:
                continue
            local_window.append(frame)
            if len(local_window) > FRAMES_PER_BATCH:
                local_window.pop(0)
            if len(local_window) >= FRAMES_PER_BATCH:
                try:
                    buffers = create_buffers(local_window, buffer_size=BUFFER_SIZE)
                except Exception as e:
                    emitir_evento("error", {"mensaje": f"bufferizer fallo: {e}"})
                    continue
                if not buffers:
                    continue
                telemetry["buffers_produced"] += 1
                # timestamp del buffer (UTC ISO)
                buffer_ts = datetime.datetime.utcnow().isoformat() + "Z"
                try:
                    # se encola un dict con buffers y timestamp
                    buffer_queue.put_nowait({"buffers": buffers, "timestamp": buffer_ts})
                except queue.Full:
                    telemetry["buffers_dropped_full"] += 1

    # Predictor
    def predictor_worker(worker_id):
        nonlocal telemetry
        global cooldown_until, last_fall_time, consecutive_detections
        while not stop_threads.is_set() and running:
            try:
                item = buffer_queue.get(timeout=1.0)
            except Exception:
                continue

            # item ahora es dict {"buffers": buffers, "timestamp": ts}
            buffers = item.get("buffers") if isinstance(item, dict) else item
            buffer_ts = item.get("timestamp") if isinstance(item, dict) else None

            # Si estamos en cooldown, descartamos los buffers (los dejamos pasar)
            now = time.time()
            if now < cooldown_until:
                telemetry["buffers_dropped_cooldown"] = telemetry.get("buffers_dropped_cooldown", 0) + 1
                continue

            # --- NUEVO: comprobar movimiento mínimo en los buffers antes de predecir ---
            try:
                motions = []
                for b in buffers:
                    if hasattr(b, "shape") and b.shape[0] >= 2:
                        # Los frames vienen normalizados [0,1] del modelo
                        diffs = np.abs(b[1:] - b[:-1])
                        mean_diff = float(np.mean(diffs))
                        motions.append(mean_diff)
        
                mean_motion = float(np.mean(motions)) if motions else 0.0

                if mean_motion < DEFAULTS["MIN_MOTION"]:
                    telemetry["buffers_dropped_no_motion"] = telemetry.get("buffers_dropped_no_motion", 0) + 1
                    continue
            except Exception as e:
                emitir_evento("error", {"mensaje": f"Motion check error: {str(e)}"})
                continue

            telemetry["buffers_processed"] += 1
            try:
                result = detector.predict_video(buffers)
            except Exception as e:
                emitir_evento("error", {"mensaje": f"Worker {worker_id} predict fallo: {e}"})
                continue

            # Modificar la sección de detección
            porcentaje = float(result.get("porcentaje", 0.0))
            if porcentaje >= MIN_DETECTION_PERCENTAGE:
                consecutive_detections += 1
                emitir_evento("deteccion_debug", {
                    "consecutivas": consecutive_detections,
                    "requeridas": DEFAULTS["MIN_CONSECUTIVE_DETECTIONS"],
                    "porcentaje": porcentaje
                })
                
                if consecutive_detections >= DEFAULTS["MIN_CONSECUTIVE_DETECTIONS"]:
                    now = time.time()
                    with lock:
                        if now >= cooldown_until:
                            last_fall_time = now
                            cooldown_until = now + cooldown_seconds
                            consecutive_detections = 0  # reiniciar contador
                            
                            # Vaciar buffers pendientes
                            cleared = 0
                            try:
                                while not buffer_queue.empty():
                                    buffer_queue.get_nowait()
                                    cleared += 1
                            except Exception:
                                pass
                                
                            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
                            emitir_evento("caida_detectada", {
                                "camara": current_camera,
                                "timestamp": timestamp,
                                "porcentaje": porcentaje,
                                "buffers_totales": result.get("buffers_totales", 0),
                                "buffer_ts": buffer_ts,
                                "buffers_cleared": cleared,
                                "detecciones_consecutivas": DEFAULTS["MIN_CONSECUTIVE_DETECTIONS"]
                            })
            else:
                consecutive_detections = 0  # reiniciar si no detecta
                emitir_evento("resultado_descartado", {
                    "camara": current_camera,
                    "porcentaje": porcentaje,
                    "buffers_totales": result.get("buffers_totales", 0),
                    "buffer_ts": buffer_ts
                })

    # Bucle principal
    while running:
        if not camera_ready or not current_rtsp:
            time.sleep(2)
            continue

        stop_threads.clear()

        reader = threading.Thread(target=start_reader, args=(current_rtsp,), daemon=True)
        reader.start()
        buf_thread = threading.Thread(target=bufferizer, daemon=True)
        buf_thread.start()
        workers = []
        for i in range(NUM_PREDICT_WORKERS):
            w = threading.Thread(target=predictor_worker, args=(i,), daemon=True)
            w.start()
            workers.append(w)

        # Mostrar ventana OpenCV en desarrollo
        try:
            while running and camera_ready:
                with frame_queue.mutex:
                    last_frame = frame_queue.queue[-1] if len(frame_queue.queue) > 0 else None
                if last_frame is not None and not getattr(sys, 'frozen', False):
                    resized = cv2.resize(last_frame, display_size)
                    cv2.imshow(f"Vista de cámara ({current_camera})", resized)
                    cv2.resizeWindow(f"Vista de cámara ({current_camera})", *display_size)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        running = False
                        break
                time.sleep(0.05)
        finally:
            stop_threads.set()
            reader.join(timeout=1.0)
            buf_thread.join(timeout=1.0)
            for w in workers:
                w.join(timeout=1.0)
            if not getattr(sys, 'frozen', False):
                cv2.destroyAllWindows()
            time.sleep(2)

# ==========================
# COMANDOS DESDE ELECTRON
# ==========================
def procesar_comando(linea):
    global current_rtsp, current_camera, camera_ready, cooldown_seconds
    try:
        cmd = json.loads(linea)
        comando = cmd.get("comando")
        if comando == "actualizar_camara":
            nueva_url = cmd.get("url", "")
            nuevo_nombre = cmd.get("nombre", "CAM-DESCONOCIDA")
            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
            cfg["LIVE_CAMERA_URL"] = nueva_url
            cfg["CAMERA_NAME"] = nuevo_nombre
            with open(CONFIG_PATH, 'w') as f:
                json.dump(cfg, f, indent=2)
            current_rtsp = nueva_url
            current_camera = nuevo_nombre
            camera_ready = bool(nueva_url.strip())
            emitir_evento("camara_actualizada", {"url": nueva_url, "nombre": nuevo_nombre})
        elif comando == "actualizar_parametros":
            nuevos_workers = int(cmd.get("predict_workers", DEFAULTS["PREDICT_WORKERS"]))
            nueva_deteccion = float(cmd.get("min_detection", DEFAULTS["MIN_DETECTION_PERCENTAGE"]))
            nuevo_cooldown = int(cmd.get("cooldown", cooldown_seconds))
            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
            cfg["PREDICT_WORKERS"] = nuevos_workers
            cfg["MIN_DETECTION_PERCENTAGE"] = nueva_deteccion
            cfg["COOLDOWN_SECONDS"] = nuevo_cooldown
            with open(CONFIG_PATH, 'w') as f:
                json.dump(cfg, f, indent=2)
            DEFAULTS["PREDICT_WORKERS"] = nuevos_workers
            DEFAULTS["MIN_DETECTION_PERCENTAGE"] = nueva_deteccion
            cooldown_seconds = nuevo_cooldown
            emitir_evento("parametros_actualizados", {
                "predict_workers": nuevos_workers,
                "min_detection": nueva_deteccion,
                "cooldown": nuevo_cooldown
            })
    except Exception as e:
        emitir_evento("error", {"mensaje": f"Error procesando comando: {str(e)}"})

def loop_escucha():
    for linea in sys.stdin:
        procesar_comando(linea.strip())

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    emitir_evento("modelo_iniciando")
    if not cargar_config_local():
        emitir_evento("error", {"mensaje": "No se pudo cargar config_local.json"})
        sys.exit(1)
    inicializar_modelo()

    # Hilo de detección
    threading.Thread(target=detectar_caidas, daemon=True).start()

    # Intento de abrir cámara en desarrollo
    if current_rtsp.strip() != "":
        def test_camera():
            global camera_ready
            cap = cv2.VideoCapture(current_rtsp, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                emitir_evento("advertencia", {"mensaje": f"No se pudo abrir RTSP: {current_rtsp}"})
                camera_ready = False
                return
            start_time = time.time()
            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if ret and frame is not None:
                    camera_ready = True
                    # emitir_evento("camara_activa", {"camara": current_camera, "rtsp": current_rtsp})
                    break
                time.sleep(0.05)
            cap.release()
            if not camera_ready:
                emitir_evento("advertencia", {"mensaje": "No se recibió ningún frame válido de la cámara"})
        threading.Thread(target=test_camera, daemon=True).start()

    loop_escucha()
