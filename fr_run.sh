#!/bin/bash
#SBATCH -p batch
#SBATCH -A CSC547
#SBATCH -t 00:10:00

PROJ_NAME="csc569"
WRKSPC="/lustre/orion/$PROJ_NAME/scratch/$USER/moe/communication"
VENV_NAME="comm-venv"


module load cray-mpich/8.1.31
module load amd-mixed/6.2.4
module load cpe/24.11
module load craype-accel-amd-gfx90a
module load cray-python/3.10.10
source $WRKSPC/$VENV_NAME/bin/activate


## calculating the number of nodes and GPUs
export NNODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=8 ## change as per your machine
export GPUS=$(( NNODES * GPUS_PER_NODE )) 

# export NNODES=1
# export GPUS_PER_NODE=1
# export GPUS=1

## pytorch dist variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS

## some RCCL env variables
# export FI_CXI_ATS=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL="PHB"

## RCCL plugin
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$WRKSPC/aws-ofi-rccl/lib"

#:$CRAY_LD_LIBRARY_PATH:${MPICH_DIR}/lib:${CRAY_MPICH_ROOTDIR}/gtl/lib"


# mpich gpu support
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_VERBOSE=1

export MPICH_OFI_NIC_POLICY="USER"
export MPICH_OFI_NIC_MAPPING="0:0-1; 1:2-3; 2:4-5; 3:6-7"

MASK_0="0x00fe000000000000" # Cores 49-55
MASK_1="0xfe00000000000000" # Cores 57-64
MASK_2="0x0000000000fe0000" # Cores 17-23
MASK_3="0x00000000fe000000" # Cores 25-31
MASK_4="0x00000000000000fe" # Cores 1-7
MASK_5="0x000000000000fe00" # Cores 9-15
MASK_6="0x000000fe00000000" # Cores 33-39
MASK_7="0x0000fe0000000000" # Cores 41-47

CPU_MASK="--cpu-bind=mask_cpu:${MASK_0},${MASK_1},${MASK_2},${MASK_3},${MASK_4},${MASK_5},${MASK_6},${MASK_7}"

# # chatgpt-o1 suggested
# Make Rendezvous kick in at 4 KiB
#export FI_CXI_RDZV_THRESHOLD=1
# Send a slightly larger eager chunk in Rendezvous
# export FI_CXI_RDZV_EAGER_SIZE=8192
# # Allow more outstanding large messages
# export FI_CXI_DEFAULT_TX_SIZE=8192
# # Keep hardware matching unless NIC resources become exhausted, then go hybrid
# export FI_CXI_RX_MATCH_MODE=hybrid
#export FI_EFA_USE_DEVICE_RDMA=1

export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0 
export MPICH_OFI_CXI_COUNTER_VERBOSE=1

export MPICH_GPU_ALLREDUCE_USE_KERNEL=1
export MPICH_GPU_ALLREDUCE_BLK_SIZE=134217728
export MPICH_GPU_ALLREDUCE_KERNEL_THRESHOLD=1
#export MPICH_OFI_RMA_STARTUP_CONNECT=1

# export FI_CXI_OFLOW_BUF_SIZE=1073741824
# export FI_CXI_OFLOW_BUF_COUNT=1

# export MPICH_SCATTERV_MIN_COMM_SIZE=1
# export MPICH_SCATTERV_SHORT_MSG=0
# export MPICH_GATHERV_MIN_COMM_SIZE=1
# export MPICH_GATHERV_SHORT_MSG=0


# - Cray MPICH supports creating a full connection grid during MPI_Init.
#         By default, OFI connections between ranks are set up on demand. This
#         allows for optimal performance while minimizing memory requirements.
#         However, for jobs requiring an all-to-all communication pattern, it
#         may be beneficial to create all OFI connections in a coordinated
#         manner at startup. See the MPICH_OFI_STARTUP_CONNECT description in
#         the mpi man page0.

#       - Cray MPICH supports runtime switching to the UCX netmod starting
#         with version 8.0.14. To do this load the craype-network-ucx module
#         and module swap between Cray-MPICH and Cray-MPICH-UCX modules.  For
#         more information including relevant environment variables reference
#         the intro_mpi man page with the Cray-MPICH-UCX module loaded.


# export CUDA_VISIBLE_DEVICES="1,0,3,2,4,5,6,7"
# CPU_MASK="--cpu-bind=mask_cpu:${MASK_1},${MASK_0},${MASK_3},${MASK_2},${MASK_4},${MASK_5},${MASK_6},${MASK_7}"

export HSA_ENABLE_SDMA=0
export MPICH_GPU_IPC_THRESHOLD=0
export MPICH_GPU_IPC_ENABLED=1

# export MPICH_GPU_IPC_ENABLED=CMA #XPMEM is the other option


SCRIPT="python -u benchmark_all_gather.py --num-gpus-per-node 8 --machine frontier --method $1"
#SCRIPT="python -u test_put.py"
#SCRIPT="python -u benchmark_reduce_scatter.py --num-gpus-per-node 8 --machine frontier --method $1"
#SCRIPT="python -u benchmark_all_to_all.py"

#SCRIPT="python -u test_put.py"


export PYTHONPATH="$PYTHONPATH:."
export MPICH_OFI_CXI_COUNTER_REPORT=5
run_cmd="srun -N $NNODES -n $GPUS --ntasks-per-node=8 -c 7 ${CPU_MASK} --mem-bind=map_mem:3,3,1,1,0,0,2,2  $SCRIPT" 
echo $run_cmd 
eval $run_cmd 



