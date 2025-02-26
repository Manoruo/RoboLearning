b=20000
r=0.02
python ../rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
  --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $r -rtg \
  --exp_name parallel_q2_b${b}_r${r} --no_gpu --p