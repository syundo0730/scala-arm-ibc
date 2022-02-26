#!/bin/bash

python3 collect_oracle.py --alsologtostderr \
  --gin_file=configs/oracle/all_states.gin \
  --serial_port=/dev/ttyUSB0 \
  --controller_serial_port=/dev/ttyUSB1 \
  --dataset_path=data/scala_arm_pixels/scala_arm_pixels*.tfrecord