#!/bin/bash
#SBATCH -p batch
#SBATCH -A CSC547
#SBATCH -t 00:30:00

PROJ_NAME="csc569"
WRKSPC="/lustre/orion/$PROJ_NAME/scratch/$USER/moe"
VENV_NAME="venv"

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
# export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL="PHB"

## RCCL plugin
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$WRKSPC/aws-ofi-rccl/lib"

# mpich gpu support
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY="GPU"
export MPICH_OFI_NUM_NICS=4
export MPICH_OFI_VERBOSE=1

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
export MPICH_OFI_RMA_STARTUP_CONNECT=1

# export FI_CXI_OFLOW_BUF_SIZE=1073741824
# export FI_CXI_OFLOW_BUF_COUNT=1



# - Cray MPICH supports creating a full connection grid during MPI_Init.
#         By default, OFI connections between ranks are set up on demand. This
#         allows for optimal performance while minimizing memory requirements.
#         However, for jobs requiring an all-to-all communication pattern, it
#         may be beneficial to create all OFI connections in a coordinated
#         manner at startup. See the MPICH_OFI_STARTUP_CONNECT description in
#         the mpi man page.

#       - Cray MPICH supports runtime switching to the UCX netmod starting
#         with version 8.0.14. To do this load the craype-network-ucx module
#         and module swap between Cray-MPICH and Cray-MPICH-UCX modules.  For
#         more information including relevant environment variables reference
#         the intro_mpi man page with the Cray-MPICH-UCX module loaded.



SCRIPT="python -u benchmark_all_gather.py"
#SCRIPT="python -u benchmark_reduce_scatter.py"
#SCRIPT="python -u benchmark_all_to_all.py"


SCRIPT="p2p/p2p_mpi.py"
export PYTHONPATH="$PYTHONPATH:."

run_cmd="srun -N $NNODES -n $GPUS -c7 --gpus-per-task=1 --gpu-bind=closest $SCRIPT" 
echo $run_cmd 
eval $run_cmd 



