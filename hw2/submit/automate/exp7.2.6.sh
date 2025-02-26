#!/bin/bash

# Define optimal batch size and learning rate
b=30000
r=0.02

# Run experiments with different settings
python ../rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r --no_gpu \
  --exp_name q4_b${b}_r${r}

python ../rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r -rtg --no_gpu \
  --exp_name q4_b${b}_r${r}_rtg

python ../rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r --nn_baseline --no_gpu \
  --exp_name q4_b${b}_r${r}_nnbaseline

python ../rob831/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
  --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r -rtg --nn_baseline --no_gpu \
  --exp_name q4_b${b}_r${r}_rtg_nnbaseline
