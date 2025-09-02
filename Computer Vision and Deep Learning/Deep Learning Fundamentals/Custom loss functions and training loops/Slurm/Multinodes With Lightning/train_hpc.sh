#!/bin/bash
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA version:"
nvcc --version || echo "nvcc not found"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "DeepSpeed Stage: ${DEEPSPEED_STAGE:-2}"

# module load python/anaconda-3.14
# conda activate /scrfs/storage/tp030/home/.conda/envs/control/
export HOME="/scrfs/storage/tp030/home"

# use conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0

export CUDA_HOME=$HOME/.conda/envs/control
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# Method 1: Using srun with DeepSpeed
# ==================================
uv run train_deepspeed.py --epochs 30 --batch_size 128 --model_size xlarge --deepspeed_stage 3 --offload_optimizer --offload_params        