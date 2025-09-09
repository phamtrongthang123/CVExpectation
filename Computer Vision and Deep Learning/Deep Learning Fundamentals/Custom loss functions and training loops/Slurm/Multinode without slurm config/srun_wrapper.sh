#!/bin/bash

script=${1}
total_nodes=${2}
host=${3}
node_rank_offset=${4}
gpus_per_node=${5:-4}
srun ${script} ${total_nodes} ${host} ${node_rank_offset} ${gpus_per_node}

