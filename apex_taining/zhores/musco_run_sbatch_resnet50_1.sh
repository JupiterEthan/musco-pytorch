#!/bin/bash


MODEL_NAME="resnet50_imagenet"
STATE_NAME="resnet50-19c8e357.pth"

epochs=12

grlayers_id=1
decomposition="tucker2"
rank_selection="vbmf"

RUNFILE_DIR="/trinity/home/y.gusak/musco/rebuttal"
SBATCH_LOGDIR="${RUNFILE_DIR}/zhores/sbatch_logs"

for factor in 0.8 0.6 1.
do
    sbatch -N 1 -p gpu_big --exclusive  -D $SBATCH_LOGDIR ${RUNFILE_DIR}/musco_run.sh \
            -m $MODEL_NAME \
            -s $STATE_NAME \
            -g $grlayers_id \
            -d $decomposition \
            -r $rank_selection \
            -f $factor \
            -i $epochs \
            -z
done

