#!/bin/bash
#SBATCH --job-name=quick_train_classification
#SBATCH --time=01:00:00
#SBATCH --output=logs-het/het.log
#SBATCH --error=logs-het/het-err.log
#SBATCH --distribution=cyclic

# Heterogeneous job component 1: A cluster
#SBATCH --nodes=2
#SBATCH --partition=A
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=public&1a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64

# Heterogeneous job component 2: BB cluster
#SBATCH hetjob
#SBATCH --partition=BB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=csce&4a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64

echo "Heterogeneous job leader (component 0) starting on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"

SCRIPT_TRAINING=train_apptainer.sh 
srun --het-group=0 --nodes 2 --ntasks-per-node=1 bash -c "bash $SCRIPT_TRAINING 3 $(hostname) 0 1" &
srun --het-group=1 --nodes 1 --ntasks-per-node=1  bash -c "bash $SCRIPT_TRAINING 3 $(hostname) 2 4" &

wait
echo "Done printing hostnames."