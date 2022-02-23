import os
import sys

import cv2
import gin


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from camera.image import apply_crop_and_reshape
from camera.video_capture import open_video_capture


def _show_modified_video():
    with open_video_capture(0) as cap:
        while True:
            ret, img = cap.read()
            cv2.imshow('original', img)
            modified = apply_crop_and_reshape(img)
            cv2.imshow('crop_and_reshaped', modified)
            cv2.waitKey(1)


def main():
    gin.parse_config_file('configs/oracle/all_states.gin', skip_unknown=True)
    _show_modified_video()


if __name__ == '__main__':
    main()
