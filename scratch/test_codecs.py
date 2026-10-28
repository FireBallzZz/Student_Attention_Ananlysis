import cv2
import numpy as np

for codec in ['mp4v', 'avc1', 'H264', 'X264']:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    filename = f"scratch/test_{codec}.mp4"
    vw = cv2.VideoWriter(filename, fourcc, 10.0, (640, 480))
    if vw.isOpened():
        # write 10 dummy frames
        for _ in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            vw.write(frame)
        vw.release()
        print(f"Codec {codec}: SUCCESS, file size {cv2.os.path.getsize(filename)} bytes")
    else:
        print(f"Codec {codec}: FAILED to open")
