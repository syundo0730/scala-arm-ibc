#!/bin/bash

python3 train_eval.py --alsologtostderr \
  --gin_file=configs/scala_arm_pixels/train_pixel_ebm.gin \
  --serial_port=/dev/ttyUSB0