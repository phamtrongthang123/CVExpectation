```bash
bash bash.sh
```
Trong file bash.sh đấy là syntax để slurm tự handle kiếm nodes cho mình bằng combo constraint + partition. Nhìn gọn và cần thủ công tùy lúc, nhưng nhìn chung sẽ giúp ích long term vì nếu cần thì cứ đọc docs của slurm.

Hiện tại có điểm yếu là do mình dùng zero3 offload nên nó compile cho 1 gpu => mình không mix gpu được.
Nhưng generally combo sbatch -> srun -> apptainer -> torchrun -> real script sẽ tự động assign gpu cho mình được.

Ngoài ra thì nhớ set auto cái devices và num nodes cho chính xác. Cũng như vô hiệu hóa function validate (check .py). 

## SLURM environment 
Mình print rất nhiều environment variables vào logs để tương lai có thể tính được slurm có những flag nào để mò lại. Slurm version hiện tại là 23.11.3. Với version slurm trước 17 thì sẽ phải chạy sbatch / torchrun ở từng máy lẻ, xong connect qua IP (nhiều người dùng rendezvous endpoint cho mục đích này). Slurm version 17+ cho phép mình set srun ngay trong 1 bash script và tận dụng được nhiều built-in feature trong cùng 1 sbatch environment. Một trong những feature mình thích nhất là nếu mình cần nhiều nodes thì slurm sẽ chờ khi có thể cấp đủ thì mới chạy cho mình. Nó hành xử như một job. Ngoài ra thì nó share chung 1 log file.

## tricky part with task distribution

### cyclic distribution
Ví dụ mình có 4 gpus nhưng 1 task cho 1 máy => fine. 
Nhưng có 4 nodes, mỗi node có 4 gpus, nhưng có 4 task => by default script hiện tại sẽ phân phát cả 4 tasks cho node đầu tiên vì nó đếm thấy có 4 gpus. Nó conflict với torchelastic (tự tăng gpu lên) => OOM hoặc nhiều job 1 gpu device. 

Một cách fix là dùng distributed cyclic 
Ví dụ 3 nodes 9 tasks sẽ phân phối như dưới
```bash
#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks=9
#SBATCH --distribution=cyclic

# Tasks 0,3,6 go to node 1
# Tasks 1,4,7 go to node 2  
# Tasks 2,5,8 go to node 3
```
Và nếu là 4 nodes 4 tasks thì sẽ phân phối đều ra: 
```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --distribution=cyclic

# Tasks 0 go to node 1
# Tasks 1 go to node 2  
# Tasks 2 go to node 3
# Tasks 3 go to node 4
```
tuy nhiên nếu máy chỉ có 1 gpu thì sẽ không bị vấn đề gì.

### srun het-group nodes ntasks-per-node
Có lưu ý là combo --nodes và --ntasks-per-node sẽ bổ trợ --ntasks nếu không thì cyclic không biết phân bổ đi đâu => vẫn bị vấn đề overlap. Combo không có conflict hiện tại là cyclic đi chung với define --nodes và --ntasks-per-node ngay srun. 
Có thể hiểu sbatch là để xin một lượng tài nguyên vô 1 pool, xong srun là để pick nhỏ ra để sử dụng. Nên cần define 2 lần.
Ví dụ đây là case mà buộc phải set --nodes và --ntasks-per-node tại srun thay vì chỉ --ntasks thì mới work. 
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
