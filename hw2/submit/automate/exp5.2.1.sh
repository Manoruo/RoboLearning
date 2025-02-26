#!/bin/bash

# Define batch sizes and learning rates to search over
BATCH_SIZES=(100 500 1000 2000 5000 10000 20000 50000)
LEARNING_RATES=(0.01 0.1 0.15 0.2 0.5) 

# Loop over batch sizes and learning rates
for b in "${BATCH_SIZES[@]}"; do
  for r in "${LEARNING_RATES[@]}"; do
    python ../rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
      --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $r -rtg \
      --exp_name q2_b${b}_r${r} --no_gpu
  done
done
