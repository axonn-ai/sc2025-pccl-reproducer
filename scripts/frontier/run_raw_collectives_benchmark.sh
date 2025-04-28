#!/bin/bash
#SBATCH -p batch
#SBATCH -A CSC547
#SBATCH -t 00:15:00

PROJ_NAME="csc547"
export WRKSPC=/lustre/orion/$PROJ_NAME/scratch/$USER/
VENV_NAME="pccl-venv"

module load cray-mpich/8.1.31
module load amd-mixed/6.2.4
module load cpe/24.11
module load craype-accel-amd-gfx90a
module load cray-python/3.10.10
module load craype-accel-amd-gfx90a
module load rocm
module load ninja
export CXX=CC 
export CC=cc

source $WRKSPC/$VENV_NAME/bin/activate

## calculating the number of nodes and GPUs
export NNODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=8 ## change as per your machine
export GPUS=$(( NNODES * GPUS_PER_NODE )) 

## pytorch dist variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS

## some RCCL env variables
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL="PHB"
## RCCL plugin
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$WRKSPC/aws-ofi-rccl/lib"

# mpich gpu support
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_VERBOSE=1
export MPICH_OFI_NIC_POLICY="USER"
export MPICH_OFI_NIC_MAPPING="0:0-1; 1:2-3; 2:4-5; 3:6-7"

# fi variables
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0 
export MPICH_OFI_CXI_COUNTER_VERBOSE=1
export MPICH_OFI_CXI_COUNTER_REPORT=5
export HSA_ENABLE_SDMA=0


MASK_0="0x00fe000000000000" # Cores 49-55
MASK_1="0xfe00000000000000" # Cores 57-64
MASK_2="0x0000000000fe0000" # Cores 17-23
MASK_3="0x00000000fe000000" # Cores 25-31
MASK_4="0x00000000000000fe" # Cores 1-7
MASK_5="0x000000000000fe00" # Cores 9-15
MASK_6="0x000000fe00000000" # Cores 33-39
MASK_7="0x0000fe0000000000" # Cores 41-47
CPU_MASK="--cpu-bind=mask_cpu:${MASK_0},${MASK_1},${MASK_2},${MASK_3},${MASK_4},${MASK_5},${MASK_6},${MASK_7}"

# collecting counter data
#export MPICH_OFI_CXI_COUNTER_REPORT=5

SCRIPT="python -u benchmark_raw_collectives/all_gather.py \
        --num-gpus-per-node $GPUS_PER_NODE \
        --machine perlmutter \
        --library pccl --test"


export PYTHONPATH="$PYTHONPATH:."


run_cmd="srun -N $NNODES -n $GPUS --ntasks-per-node=8 -c 7 ${CPU_MASK} --mem-bind=map_mem:3,3,1,1,0,0,2,2  bash -c 'ulimit -c 0; exec $SCRIPT'"
echo $run_cmd 
eval $run_cmd 



