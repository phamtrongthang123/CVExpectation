```bash
sbatch --nodelist c1912,c2008 --partition agpu --job-name ita-wan-near-deadline --output slurm-logs/exp-2nodes-2GPUS-ita-wan-near-deadline/log-ita-wan-near-deadline-mrank0-agpu.log --error slurm-logs/exp-2nodes-2GPUS-ita-wan-near-deadline/err-ita-wan-near-deadline-mrank0-agpu.log --ntasks 2 --ntasks-per-node 1 --gres gpu:1 --cpus-per-task 64 --time 01:00:00  /scrfs/storage/tp030/home/ftvae/ita-mdt_code_wan_vae/multinode/srun_wraper.sh train_apptainer.sh 3 c1912 0 1

sbatch --nodelist c2104 --partition condo --job-name ita-wan-near-deadline --output slurm-logs/exp-2nodes-2GPUS-ita-wan-near-deadline/log-ita-wan-near-deadline-mrank1-condo.log --error slurm-logs/exp-2nodes-2GPUS-ita-wan-near-deadline/err-ita-wan-near-deadline-mrank1-condo.log --ntasks 1 --ntasks-per-node 1 --gres gpu:4 --cpus-per-task 64 --time 01:00:00  /scrfs/storage/tp030/home/ftvae/ita-mdt_code_wan_vae/multinode/srun_wraper.sh train_apptainer.sh 3 c1912 2 4
```
Hiện tại có điểm yếu là do mình dùng zero3 offload nên nó compile cho 1 gpu => mình không mix gpu được.
Nhưng generally combo sbatch -> srun -> apptainer -> torchrun -> real script sẽ tự động assign gpu cho mình được.

Ngoài ra thì nhớ set auto cái devices và num nodes cho chính xác. Cũng như vô hiệu hóa function validate (check .py). 
