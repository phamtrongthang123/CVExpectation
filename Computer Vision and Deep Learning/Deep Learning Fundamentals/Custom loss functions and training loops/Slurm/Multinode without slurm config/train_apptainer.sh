#!/bin/bash
# TorchElastic distributed training arguments
TOTAL_NODES=${1}
MASTER_ADDR=${2}
OFFSET_RANK=${3}

export HOME="/scrfs/storage/tp030/home"
export OMP_NUM_THREADS=24

export NCCL_DEBUG=INFO

# Run TorchElastic training with Apptainer container
# torchrun manages the distributed processes and calls the container for each process
apptainer exec --nv --writable-tmpfs \
        --env VLLM_SKIP_P2P_CHECK=1 \
        --bind /scrfs/storage/tp030/home:/scrfs/storage/tp030/home \
        $HOME/qwenvl-2.5-cu121.sif bash train.sh $TOTAL_NODES $MASTER_ADDR $OFFSET_RANK 
