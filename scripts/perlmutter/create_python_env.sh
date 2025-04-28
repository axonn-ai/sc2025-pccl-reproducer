#!/bin/bash
#
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

WRKSPC=$SCRATCH
# everything will be installed in $WRKSPC

ENV_NAME="pccl-venv"
# this is the name of your python venv, change if needed

cd $WRKSPC
echo -e "${RED}Creating Python Environment in $WRKSPC:${GREEN}"
module load python 
python -m venv $WRKSPC/$ENV_NAME 
module unload python

echo -e "${RED}Installing Dependencies:${GREEN}"

#Step 1 - activate your venv
source $WRKSPC/$ENV_NAME/bin/activate

#Step 2 - install pip packages
pip install --no-cache-dir  --upgrade pip
pip install --no-cache-dir  torch torchvision
pip install --no-cache-dir  deepspeed
pip install --no-cache-dir  transformers datasets tiktoken wandb tqdm

# Step 3 - install mpi4py over cray-mpich
module load PrgEnv-gnu cray-mpich craype-accel-nvidia80
MPICC="cc -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

echo -e "${RED}Your Python Environment is ready. To activate it run the following commands in the SAME order:${NC}"
echo -e "${GREEN}source $WRKSPC/$ENV_NAME/bin/activate${NC}"
echo ""
echo -e "${NC}"