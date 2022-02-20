#!/bin/bash

python3 policy_eval.py --alsologtostderr \
  --gin_file=configs/run_mlp_ebm.gin \
  --serial_port=/dev/tty.usbserial-1110 \
  --saved_model_path=/tmp/ibc_logs/mlp_ebm/ibc_dfo/20220213-034118/policies/greedy_policy \
  --checkpoint_path=/tmp/ibc_logs/mlp_ebm/ibc_dfo/20220213-034118/policies/checkpoints/policy_checkpoint_0000020000
