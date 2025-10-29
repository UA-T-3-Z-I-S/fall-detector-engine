import cv2

def load_rtsp_frames(rtsp_url, num_frames=32, resize_to=(224, 224)):
    """
    Captura N frames consecutivos desde un stream RTSP.
    Devuelve una lista de frames redimensionados.
    """
    cap = cv2.VideoCapture(rtsp_url)
    frames = []
    if not cap.isOpened():
        print(f"[❌] No se pudo abrir el stream RTSP: {rtsp_url}")
        return frames

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_to:
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()
    return frames
