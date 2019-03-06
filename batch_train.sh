#!/usr/bin/env bash
for SEED in {1..10}
do
   ./train.py -c configs/train/cartpole/dqn2015.conf --seed $SEED --wandb
done
