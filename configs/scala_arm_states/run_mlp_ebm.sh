#!/bin/bash

python3 train_eval.py --alsologtostderr \
  --gin_file=configs/scala_arm_states/mlp_ebm.gin \
  --add_time=True \
  --env_name=ScalaArm-v0
