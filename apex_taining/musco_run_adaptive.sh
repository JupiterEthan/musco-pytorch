#!/bin/bash    
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH  --gpus-per-node=4
#SBATCH --exclusive     
#SBATCH --partition gpu_big


# Parse short options
while getopts m:s:g:d:r:i:z option
do
case "${option}"
in
m) MODEL_NAME=${OPTARG};;
s) MODEL_STATE=${OPTARG};;
g) grschedule_id=${OPTARG};;
d) decomposition=${OPTARG};;
r) rank_selection=${OPTARG};;
i) EPOCHS=${OPTARG};;
z) zhores=1;;
esac
done

# Add packages while working on zhores
if [[ $zhores ]]
then
    module load compilers/gcc-5.5.0 
    module load gpu/cuda-10.0
    module load python/python-3.6.8

    pip install --upgrade pip --user
    pip install  absl-py --user

    #pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl --user
    #pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl --user

    pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl --user
    pip3 install torchvision==0.2.1 --user


    pip install Cython --user
    #pip install maxvolpy --user
    pip install catalyst --user
    pip install  scikit-tensor-py3 --user
    pip install tensorly --user

    if [ ! -d /trinity/home/y.gusak/apex/.git ]
    then
        cd /trinity/home/y.gusak/ && git clone https://github.com/NVIDIA/apex
    fi

    cd /trinity/home/y.gusak/apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd /trinity/home/y.gusak/
fi


echo 'HOOOOOOOOOOOOOOOOOOOO'

username="y.gusak"
# MODEL_NAME="resnet18_imagenet"
# grschedule_id=0
# decomposition="cp3"
# rank_selection="nx"
# factor="2.2"

DATA_DIR="/gpfs/gpfs0/e.ponomarev/imagenet"
src_dir="/trinity/home/y.gusak/musco/rebuttal"

COMPRESS_SCRIPT="$src_dir/compress_subiter.py"
EVAL_SCRIPT="$src_dir/eval_compressed_subiter.py"

FINETUNE_SCRIPT="$src_dir/finetune_iter.py"


torchvisionmodels_dir="/gpfs/gpfs0/y.gusak/pretrained/torchvision/models_torch_1-0-1"
INITIALMODEL_PATH="${torchvisionmodels_dir}/$MODEL_NAME.pth"

torchvisionstates_dir="/gpfs/gpfs0/y.gusak/pretrained/torchvision/state_dicts"
INITIALSTATE_PATH="${torchvisionstates_dir}/$MODEL_STATE"

muscomodels_dir="/gpfs/gpfs0/y.gusak/musco_models/${MODEL_NAME}"

GRSCHEDULE_DIR="${muscomodels_dir}/grschedule${grschedule_id}"
mkdir -p $GRSCHEDULE_DIR

fname="${GRSCHEDULE_DIR}.txt"

initial_prec1=92.0
delta_prec1=10.0

if [[ $rank_selection == "vbmf" ]]
then
    suffix="wf"
else
    suffix="xf"
fi
    
echo 'Suffix' $suffix 'adaptive'

if [[ $grschedule_id == 0 ]]
then
    maxlocaliter=3
elif [[ $grschedule_id == 1 ]]
then
    maxlocaliter=7
elif [[ $grschedule_id == 1000 ]]
then
    maxlocaliter=15    
fi

echo 'Maxlocaliter' $maxlocaliter
    

GLOBAL_DIR="${GRSCHEDULE_DIR}/${decomposition}_${rank_selection}_${suffix}-adaptive"
mkdir -p $GLOBAL_DIR


for globaliter in 0 1 2
do
    echo 'Global iter' $globaliter
    localiter=0
    
    cat $fname | while read lnames || [ -n "$lnames" ]
    do   
        localdir="${GLOBAL_DIR}/iter_${globaliter}-${localiter}"
        mkdir -p $localdir
        
        pylogsdir="$localdir/pylogs"
        mkdir -p $pylogsdir
        
        echo 'Local iter' $localiter ', layers to compress:' $lnames
        echo $localdir
        
        echo 'Start compression'
        
        wfs=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
        lenwfs=${#wfs[@]}
        
        for (( i=0; i<$lenwfs; i++))
        do
            fact="${wfs[$i]}"
            one=1

            if [[ $i == `expr $lenwfs - $one` ]]
            then
                is_largest_fact="--is-largest-wf"
                echo 'Largest fact' $fact
            else
                is_largest_fact=''
            fi

            echo 'Wf' $fact
            bash -c "cd ~ && CUDA_VISIBLE_DEVICES=''\
                    python3 $COMPRESS_SCRIPT \
                    --global_dir $GLOBAL_DIR \
                    --global_iter $globaliter \
                    --local_iter $localiter \
                    --max_local_iter $maxlocaliter \
                    --initialmodel_path $INITIALMODEL_PATH \
                    --initialstate_path $INITIALSTATE_PATH \
                    --lnames '$lnames' \
                    --decomposition $decomposition \
                    --rank_selection $rank_selection \
                    --factor $fact  >>'$pylogsdir/compress_logs.log' 2>'$pylogsdir/compress_errors.log'"
                    
            echo 'Calibrate BN'
            bash -c "cd ~ && CUDA_DEVICE_ORDER=PCI_BUS_ID \
                CUDA_VISIBLE_DEVICES=0,1,2,3
                python3 -m torch.distributed.launch \
                --nproc_per_node=4 $EVAL_SCRIPT\
                -a $MODEL_NAME \
                --b 128 \
                --workers 16 \
                --opt-level O1 \
                --saveload_dir $localdir \
                --ftlogsdir $pylogsdir \
                --lr 0.0 \
                --epochs 1 \
                --data $DATA_DIR \
                --initial-prec1 $initial_prec1 \
                --delta-prec1 $delta_prec1 \
                --pretrained \
                $is_largest_fact>>'$pylogsdir/eval_compressed_logs.log' 2>'$pylogsdir/eval_compressed_errors.log'"
        done


#         echo 'Start fine-tuning'
#       #  rm -r ${pylogsdir}/finetune*
#         ftlogsdir="${localdir}/ftlogs"
#         mkdir -p $ftlogsdir
        
#         bash -c "CUDA_DEVICE_ORDER=PCI_BUS_ID \
#                 CUDA_VISIBLE_DEVICES=0,1,2,3
#                 python3 -m torch.distributed.launch \
#                 --nproc_per_node=4 $FINETUNE_SCRIPT \
#                 -a $MODEL_NAME \
#                 --b 128 \
#                 --workers 16 \
#                 --opt-level O1 \
#                 --saveload_dir $localdir \
#                 --ftlogsdir $ftlogsdir \
#                 --lr 0.0001 \
#                 --epochs $EPOCHS \
#                 --data $DATA_DIR \
#                 --pretrained >>'$pylogsdir/finetune_logs.log' 2>'$pylogsdir/finetune_errors.log'" 
        
       # bash -c "CUDA_DEVICE_ORDER=PCI_BUS_ID \
       #         CUDA_VISIBLE_DEVICES=0,1,2,3 \
       #         python3  $FINETUNE_SCRIPT \
       #         -a $MODEL_NAME \
       #         --b 128 \
       #         --workers 16 \
       #         --opt-level O1 \
       #         --saveload_dir $localdir \
       #         --lr 0.0001 \
       #         --epochs 15 \
       #         --data $DATA_DIR \
       #         --pretrained >>'$pylogsdir/finetune_logs.log' 2>'$pylogsdir/finetune_errors.log'"



        
        localiter=$((localiter + 1))
    done
done
