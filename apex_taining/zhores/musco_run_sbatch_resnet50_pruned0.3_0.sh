#!/bin/bash


MODEL_NAME="resnet50_pruned0.3"
STATE_NAME="resnet50_pruned0.3.pth"

epochs=13

grlayers_id=0
decomposition="tucker2"
rank_selection="vbmf"

RUNFILE_DIR="/trinity/home/y.gusak/musco/rebuttal"
SBATCH_LOGDIR="${RUNFILE_DIR}/zhores/sbatch_logs"

# 0.8 0.6 
for factor in  0.8 0.6
do
    sbatch -N 1 -p gpu_big --exclusive  -D $SBATCH_LOGDIR ${RUNFILE_DIR}/musco_run_dcp.sh \
            -m $MODEL_NAME \
            -s $STATE_NAME \
            -g $grlayers_id \
            -d $decomposition \
            -r $rank_selection \
            -f $factor \
            -i $epochs \
            -z
done

