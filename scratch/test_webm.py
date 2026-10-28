import cv2
import numpy as np

for codec, ext in [('VP80', 'webm'), ('VP90', 'webm'), ('mp4v', 'mp4')]:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    filename = f"scratch/test_{codec}.{ext}"
    vw = cv2.VideoWriter(filename, fourcc, 10.0, (640, 480))
    if vw.isOpened():
        for _ in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            vw.write(frame)
        vw.release()
        print(f"Codec {codec} ({ext}): SUCCESS, size {cv2.os.path.getsize(filename)} bytes")
    else:
        print(f"Codec {codec} ({ext}): FAILED to open")
