#!/bin/bash

# Define the values of lambda to search over
lambda_values=(0 0.95 0.99 1)

# Loop over lambda values
for lambda in "${lambda_values[@]}"; do
  # Run the Python script with the current lambda value
  python ../rob831/scripts/run_hw2.py \
    --env_name Hopper-v4 --ep_len 1000 \
    --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 \
    --reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda $lambda \
    --exp_name q5_b2000_r0.001_lambda${lambda} --no_gpu
done
