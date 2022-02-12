from contextlib import contextmanager

import cv2


@contextmanager
def open_video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        available = cap.read()[0]
        if not available:
            raise RuntimeError('camera unavailable')
        yield cap
    finally:
        cap.release()
