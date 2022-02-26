#!/bin/bash

python3 policy_eval.py --alsologtostderr \
  --gin_file=configs/run_mlp_ebm.gin \
  --serial_port=/dev/ttyUSB0 \
  --saved_model_path=/tmp/scala_arm_ibc_logs/conv_mlp_ebm/policies/greedy_policy \
  --checkpoint_path=/tmp/scala_arm_ibc_logs/conv_mlp_ebm/policies/checkpoints/policy_checkpoint_0000030000
