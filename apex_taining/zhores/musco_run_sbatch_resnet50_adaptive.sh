#!/bin/bash


MODEL_NAME="resnet50_imagenet"
STATE_NAME="resnet50-19c8e357.pth"

epochs=15

grlayers_id=1000
decomposition="tucker2"
rank_selection="vbmf"

RUNFILE_DIR="/trinity/home/y.gusak/musco/rebuttal"
SBATCH_LOGDIR="${RUNFILE_DIR}/zhores/sbatch_logs"

sbatch -N 1 -p gpu_debug --exclusive  -D $SBATCH_LOGDIR ${RUNFILE_DIR}/musco_run_adaptive.sh \
        -m $MODEL_NAME \
        -s $STATE_NAME \
        -g $grlayers_id \
        -d $decomposition \
        -r $rank_selection \
        -i $epochs \
        -z
