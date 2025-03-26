#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH -A m2404_g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --time=00:05:00
#SBATCH --qos=regular


module load nccl
module load cudatoolkit/12.4
source ./venv/bin/activate

## calculating the number of nodes and GPUs
export NNODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=4 ## change as per your machine
export GPUS=$(( NNODES * GPUS_PER_NODE )) 


## pytorch dist variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS

#export MPICH_OFI_CXI_COUNTER_REPORT=5

source ./pm_env.sh

SCRIPT="python -u $1 --num-gpus-per-node $GPUS_PER_NODE --machine perlmutter --method $2 --use-yacl"

# for JIT building
export CXX=CC 
export CC=cc

export PYTHONPATH="$PYTHONPATH:."
run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 $CPU_MASK --gpus-per-node=4 ./get_rank.sh $SCRIPT"
echo $run_cmd 
eval $run_cmd

# alternative launch commands
#run_cmd="srun -N $NNODES -n $GPUS -c8 --gpus-per-task=1 --gpu-bind=closest ./get_rank.sh python -u $SCRIPT"
#run_cmd="srun -C gpu -N $NNODES -n $GPUS --cpus-per-task=32 --gpus-per-node=4 python -u $SCRIPT"
