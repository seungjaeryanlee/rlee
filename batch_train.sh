#!/usr/bin/env bash
for SEED in {1..10}
do
   ./train.py -c configs/train/cartpole/dqn2015.conf --agent dqn2015 --seed $SEED --wandb --replay-buffer-type uniform
   # ./train.py -c configs/train/cartpole/dqn2015.conf --agent dqn2015 --seed $SEED --wandb --replay-buffer-type combined
   # ./train.py -c configs/train/cartpole/dqn2015.conf --agent doubledqn --seed $SEED --wandb --replay-buffer-type uniform
   # ./train.py -c configs/train/cartpole/dqn2015.conf --agent doubledqn --seed $SEED --wandb --replay-buffer-type combined
done
