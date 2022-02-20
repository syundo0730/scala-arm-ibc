#!/bin/bash

python3 collect_oracle.py --alsologtostderr \
  --gin_file=configs/oracle/all_states.gin \
  --serial_port=/dev/tty.usbserial-0001 \
  --controller_serial_port=/dev/tty.usbserial-11310 \
  --dataset_path=data/data_file*.tfrecord