import os

def obtener_etiqueta_real(path):
    """
    Determina si el video proviene de una carpeta de 'caida' o 'no_caida'.
    """
    partes = os.path.normpath(path).split(os.sep)
    for p in partes:
        if p.lower() == 'caida':
            return 1
        elif p.lower() == 'no_caida':
            return 0
    return None  # Desconocido
