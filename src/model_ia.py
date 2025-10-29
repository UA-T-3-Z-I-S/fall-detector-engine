import sys
import json
import os
import cv2
import threading
import datetime
import time
import numpy as np
from dotenv import load_dotenv

# === CONFIGURACIÓN BASE ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

load_dotenv(os.path.join(ROOT_DIR, '.env'))
CONFIG_LOCAL_PATH = os.path.join(ROOT_DIR, 'config_local.json')

# === IMPORTS RELATIVOS ===
from src.preprocessing.buffer_creator import create_buffers
from src.model.predictor_separated import FallDetectorSeparated

# === VARIABLES GLOBALES ===
detector = None
current_rtsp = ""
current_camera = "CAM-DESCONOCIDA"
running = True
lock = threading.Lock()
camera_ready = False
# ===========================


# ----------- FUNCIONES BÁSICAS -------------
def emitir_evento(evento, data=None):
    """Envía salida JSON (para Electron)"""
    payload = {"evento": evento}
    if data:
        payload.update(data)
    print(json.dumps(payload), flush=True)


def cargar_config_local():
    """Carga config_local.json con RTSP y nombre de cámara"""
    global current_rtsp, current_camera, camera_ready
    try:
        if not os.path.exists(CONFIG_LOCAL_PATH):
            emitir_evento("advertencia", {"mensaje": "No se encontró config_local.json. Esperando configuración inicial..."})
            camera_ready = False
            return

        with open(CONFIG_LOCAL_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            current_rtsp = cfg.get("LIVE_CAMERA_URL", "")
            current_camera = cfg.get("CAMERA_NAME", "CAM-DESCONOCIDA")

        if not current_rtsp.strip():
            emitir_evento("advertencia", {"mensaje": "No se ha configurado una cámara RTSP. Esperando configuración..."})
            camera_ready = False
        else:
            camera_ready = True
            emitir_evento("config_cargada", {"camara": current_camera, "rtsp": current_rtsp})

    except Exception as e:
        emitir_evento("error", {"mensaje": f"No se pudo cargar config_local.json: {e}"})
        camera_ready = False


def inicializar_modelo():
    """Carga el modelo CNN/LSTM (modo separado por defecto)"""
    global detector
    try:
        detector = FallDetectorSeparated()
        emitir_evento("modelo_listo", {"modo": "separado"})
    except Exception as e:
        emitir_evento("error", {"mensaje": f"Error cargando modelo: {e}"})

# ----------- DETECCIÓN POR RTSP -------------
def detectar_caidas():
    """Lee continuamente del stream RTSP y detecta caídas"""
    global running, current_rtsp, current_camera, camera_ready

    while running:
        # Esperar configuración si no hay cámara
        if not camera_ready or not current_rtsp:
            time.sleep(2)
            continue

        cap = cv2.VideoCapture(current_rtsp)

        if not cap.isOpened():
            emitir_evento("advertencia", {"mensaje": f"No se pudo acceder a la cámara RTSP: {current_rtsp}. Esperando configuración..."})
            camera_ready = False
            time.sleep(5)
            continue

        emitir_evento("camara_activa", {"camara": current_camera, "rtsp": current_rtsp})

        frames = []
        while running and camera_ready:
            ret, frame = cap.read()
            if not ret:
                emitir_evento("advertencia", {"mensaje": "Stream interrumpido. Esperando reconexión..."})
                camera_ready = False
                break

            frames.append(frame)

            # Procesa cada 32 frames → 2 buffers de 16
            if len(frames) >= 32:
                with lock:
                    buffers = create_buffers(frames, buffer_size=16)
                    if buffers:
                        result = detector.predict_video(buffers)
                        if result.get("caida"):
                            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
                            emitir_evento("caida_detectada", {
                                "camara": current_camera,
                                "timestamp": timestamp
                            })
                frames.clear()

            time.sleep(0.05)  # regula carga de CPU

        cap.release()
        time.sleep(2)  # evita reconexión rápida


# ----------- COMANDOS DESDE ELECTRON -------------
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
    """Escucha comandos desde Electron por stdin"""
    for linea in sys.stdin:
        procesar_comando(linea.strip())


# ----------- MODO SIMULACIÓN -------------
def simulador_eventos():
    """Genera eventos simulados si no hay RTSP"""
    global running, current_camera
    try:
        with open(CONFIG_LOCAL_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            sim_enabled = cfg.get("ENABLE_SIMULATION", False)
            interval = cfg.get("SIMULATION_INTERVAL", 10)
    except:
        sim_enabled = False
        interval = 10

    while running and sim_enabled:
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        emitir_evento("caida_detectada", {"camara": current_camera, "timestamp": timestamp, "simulado": True})
        time.sleep(interval)


# ----------- MAIN -------------
if __name__ == "__main__":
    emitir_evento("modelo_iniciando")

    cargar_config_local()
    inicializar_modelo()

    # Inicia hilos paralelos
    threading.Thread(target=detectar_caidas, daemon=True).start()
    threading.Thread(target=simulador_eventos, daemon=True).start()

    loop_escucha()
