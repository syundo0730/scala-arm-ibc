#!/bin/bash

python3 train_eval.py --alsologtostderr \
  --gin_file=configs/scala_arm_states/mlp_ebm.gin \
  --serial_port=/dev/tty.usbserial-1110 \
  --add_time=True
