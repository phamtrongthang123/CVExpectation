```bash
bash bash.sh
```
Hiện tại có điểm yếu là do mình dùng zero3 offload nên nó compile cho 1 gpu => mình không mix gpu được.
Nhưng generally combo sbatch -> srun -> apptainer -> torchrun -> real script sẽ tự động assign gpu cho mình được.

Ngoài ra thì nhớ set auto cái devices và num nodes cho chính xác. Cũng như vô hiệu hóa function validate (check .py). 


## tricky part
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
