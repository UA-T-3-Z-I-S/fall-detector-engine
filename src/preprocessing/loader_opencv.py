import cv2

def load_video_frames(video_path, max_frames=None, resize_to=(224, 224), skip_frames=2):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0

    if not cap.isOpened():
        print(f"[❌] No se pudo abrir el video: {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or (max_frames and len(frames) >= max_frames):
            break

        if resize_to:
            frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
        frames.append(frame)

        frame_idx += skip_frames  # Salta N frames directamente

    cap.release()
    return frames
