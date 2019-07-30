#!/bin/bash
DIR="/trinity/home/y.gusak/musco/rebuttal"

python3 -m torch.distributed.launch --nproc_per_node=4 $DIR/apex_train.py
