import os

def verificar_video_mp4(ruta_video):
    """
    Verifica si la ruta es válida para procesar un video .mp4.
    - Si es archivo: verifica que sea .mp4.
    - Si es carpeta: busca el primer .mp4 y lo devuelve.
    - Si no es válido, retorna None.
    """
    if os.path.isdir(ruta_video):
        for archivo in os.listdir(ruta_video):
            if archivo.lower().endswith('.mp4'):
                return os.path.join(ruta_video, archivo)
        print(f"[⚠️] Carpeta sin archivos .mp4: {ruta_video}")
        return None

    if not ruta_video.lower().endswith('.mp4'):
        print(f"[⚠️] Video descartado por no ser .mp4: {ruta_video}")
        return None

    return ruta_video
