Just call:
```bash
bash bash.sh
```

In the bash.sh file, there's syntax for Slurm to automatically handle finding nodes for you using a combination of constraint + partition. It looks clean and requires manual adjustment from time to time, but overall it's helpful long-term because you can just read Slurm's documentation when needed.

Currently, there's a weakness: since I'm using zero3 offload, it compiles for 1 GPU => I can't mix GPUs.

However, generally the combo of sbatch -> srun -> apptainer -> torchrun -> actual script will automatically assign GPUs for you.

Additionally, remember to set devices and num nodes automatically for accuracy. Also disable the validate function (check .py file).

## SLURM Environment

I print many environment variables to logs so in the future I can figure out what flags Slurm has for reference. Current Slurm version is 23.11.3. With Slurm versions before 17, you had to run sbatch/torchrun on each individual machine, then connect via IP (many people use rendezvous endpoint for this purpose). Slurm version 17+ allows you to set srun directly within a bash script and take advantage of many built-in features within the same sbatch environment. One of my favorite features is that if I need multiple nodes, Slurm will wait until it can allocate enough before running. It behaves like a job. Additionally, it shares a single log file.

## Apptainer container with no conda
We can still use source. 
First, find the location of conda in the main machine: 
```
conda info --base
# return /share/conda
```

Then simply (assume the env's name is pytorch3d): 
```
source /share/conda/bin/activate pytorch3d
```

## Tricky Part with Task Distribution

### Cyclic Distribution

For example, if I have 4 GPUs but 1 task per machine => fine.

But with 4 nodes, each node having 4 GPUs, but only 4 tasks => by default the current script will distribute all 4 tasks to the first node because it counts 4 GPUs available. This conflicts with torchelastic (which automatically scales up GPUs) => OOM or multiple jobs on one GPU device.

One fix is to use cyclic distribution.

For example, 3 nodes with 9 tasks will distribute as follows:

```bash
#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks=9
#SBATCH --distribution=cyclic
# Tasks 0,3,6 go to node 1
# Tasks 1,4,7 go to node 2  
# Tasks 2,5,8 go to node 3
```

And with 4 nodes and 4 tasks, it will distribute evenly:

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --distribution=cyclic
# Task 0 goes to node 1
# Task 1 goes to node 2  
# Task 2 goes to node 3
# Task 3 goes to node 4
```

However, if a machine only has 1 GPU, there won't be any issues.

### srun het-group nodes ntasks-per-node

Note that the combo of --nodes and --ntasks-per-node will complement --ntasks, otherwise cyclic doesn't know how to distribute => still has overlap issues. The non-conflicting combo currently is cyclic paired with defining --nodes and --ntasks-per-node directly at srun.

You can think of sbatch as requesting a pool of resources, then srun is for picking smaller portions to use. So you need to define twice.

For example, here's a case where you must set --nodes and --ntasks-per-node at srun instead of just --ntasks for it to work:

```bash
#!/bin/bash
#SBATCH --job-name=name
#SBATCH --time=3-00:00:00
#SBATCH -o slurm_log2/%N_1024_l12_%j.txt
#SBATCH -e slurm_log2/%N_1024_l12_%j.err
#SBATCH --distribution=cyclic
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tp030@uark.edu
# Heterogeneous job component 1: AA cluster
#SBATCH --nodes=4
#SBATCH --partition=AA
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=public&4a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# Heterogeneous job component 2: B cluster
#SBATCH hetjob
#SBATCH --partition=B
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=public2&4a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
echo "Heterogeneous job leader (component 0) starting on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
SCRIPT_TRAINING=train_apptainer.sh 
srun --het-group=0 --nodes=4 --ntasks-per-node=1 bash -c "bash $SCRIPT_TRAINING 6 $(hostname) 0" &
srun --het-group=1 --nodes=2 --ntasks-per-node=1 bash -c "bash $SCRIPT_TRAINING 6 $(hostname) 4" &
wait
echo "Done printing hostnames."
```
