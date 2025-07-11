import numpy as np
import cv2

def create_buffers(frames, buffer_size=16, overlap=0.0, target_size=(224, 224)):
    step = int(buffer_size * (1 - overlap))
    total = len(frames)
    buffers = []

    for i in range(0, total - buffer_size + 1, step):
        buffer_frames = frames[i:i + buffer_size]

        resized = np.empty((buffer_size, target_size[1], target_size[0], 3), dtype=np.float32)
        for j in range(buffer_size):
            resized[j] = cv2.resize(buffer_frames[j], target_size, interpolation=cv2.INTER_AREA) / 255.0

        buffers.append(resized)

    return buffers
