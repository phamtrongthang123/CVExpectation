#!/bin/bash
#SBATCH --job-name=deepspeed-lightning
#SBATCH --nodes=2                  # number of nodes you want to use. Pytorch lightning will automatically scale the number of GPUs based on the number of nodes defined here.
#SBATCH --ntasks-per-node=1         # you will get warning of not honor but lightning still honors it. Not sure why.        
#SBATCH --gres=gpu:1                 
#SBATCH --cpus-per-task=8 
#SBATCH --time=03:00:00
#SBATCH --partition=agpu            # agpu is the gpu partition, make sure you have the correct constraint for the gpu you want to use.
#SBATCH --constraint='1a100'        # 1a100 is the constraint for the gpu you want to use. Depends on the server, my server has "1a100" as a tag so I can use it.
#SBATCH --output=logs/deepspeed_%j.out
#SBATCH --error=logs/deepspeed_%j.err


nvidia-smi
export HOME="/scrfs/storage/tp030/home"
export OMP_NUM_THREADS=24

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0

# this is actually more complex than normal srun python script.py because we have another extra apptainer env to run the script. Thankfully it behaves quite tame and doesn't require too much setup.
srun apptainer exec --nv --writable-tmpfs --env VLLM_SKIP_P2P_CHECK=1 --bind /scrfs/storage/tp030/home:/scrfs/storage/tp030/home $HOME/qwenvl-2.5-cu121.sif bash -c "bash train_hpc.sh"