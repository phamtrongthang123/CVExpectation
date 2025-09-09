#!/bin/bash

nvidia-smi
# module load python/anaconda-3.14
# conda activate /scrfs/storage/tp030/home/.conda/envs/control/
export HOME="/scrfs/storage/tp030/home"

# use conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0
export CUDA_HOME=$HOME/.conda/envs/control
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA version:"
nvcc --version || echo "nvcc not found"

# Additional PyTorch Lightning specific variables
export PL_TORCH_DISTRIBUTED_BACKEND="nccl"
TOTAL_NODES=${1}
MASTER_ADDR=${2}
OFFSET_RANK=${3}
GPUS_PER_NODE=${4}
CURRENT_RANK=$((SLURM_PROCID + OFFSET_RANK))
MASTER_PORT=2010
INIT_METHOD="tcp://${MASTER_ADDR}:${MASTER_PORT}"

source ../.venv/bin/activate

echo 'Current Rank' ${CURRENT_RANK} 'LOCAL_RANK' ${SLURM_PROCID}
torchrun --nnodes=${TOTAL_NODES} --nproc-per-node=${GPUS_PER_NODE} --master-port=${MASTER_PORT} \
    --master-addr ${MASTER_ADDR} --node-rank=${CURRENT_RANK} \
    train_deepspeed.py
