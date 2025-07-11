import random
import os

def seleccionar_video_aleatorio():
    from config.paths import VIDEO_SOURCES

    opciones = ['caida', 'no_caida']
    clase = random.choice(opciones)  # elige una clase aleatoriamente
    carpeta = VIDEO_SOURCES['test'][clase]

    archivos = [f for f in os.listdir(carpeta) if f.endswith('.mp4')]
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos .mp4 en: {carpeta}")

    archivo = random.choice(archivos)
    ruta = os.path.join(carpeta, archivo)
    return ruta
