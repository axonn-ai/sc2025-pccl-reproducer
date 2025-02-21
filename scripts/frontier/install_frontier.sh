#!/bin/bash

module load PrgEnv-cray
rocm_version="6.2.4"
module load amd-mixed/${rocm_version}
module load cray-mpich/8.1.31
module load cpe/24.11
module load craype-accel-amd-gfx90a
module load cray-python/3.10.10
module load libtool

export ROCM_PATH="/opt/rocm-${rocm_version}/"

LITGPT_DIR=$(pwd)

PROJ_NAME="csc569"
export WRKSPC=/lustre/orion/$PROJ_NAME/scratch/$USER/moe/communication
mkdir -p $WRKSPC
cd $WRKSPC
ENV_NAME="comm-venv"
ENV_LOC="$WRKSPC/$ENV_NAME"

# Setup Virtual Environment
echo "Setting up Virtual Environment"
python -m venv ${ENV_LOC} --system-site-packages
. ${ENV_LOC}/bin/activate

pip install --upgrade pip

# PyTorch
echo "Installing PyTorch"
if [ "${rocm_version}" == 5.6.0  ]; then
	pip install --force-reinstall /lustre/orion/world-shared/stf007/msandov1/wheels/TorchROCm5.6/torch-2.1.2-cp310-cp310-linux_x86_64.whl
elif [ "${rocm_version}" == 6.0.0  ]; then
	pip3 install torch  --index-url https://download.pytorch.org/whl/rocm6.0
elif [ "${rocm_version}" == 5.7.0  ]; then
	pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/rocm5.7
elif [ "${rocm_version}" == 6.2.4  ]; then
	pip3 install torch --index-url https://download.pytorch.org/whl/rocm6.2.4
	pip install --upgrade numpy
fi

# mpi4py
export MPICH_GPU_SUPPORT_ENABLED=1
INC="-I${ROCM_PATH}/include"
LDFLAGS="-L${ROCM_PATH}/lib -lamdhip64"
#CC=cc
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
MPICC="cc -shared" INC=$INC LDFLAGS=$LDFLAGS pip install --upgrade --no-cache-dir --no-binary=mpi4py mpi4py

# # Flash Attention
# echo "Installing Flash Attention"
# git clone https://github.com/ROCmSoftwarePlatform/flash-attention
# cd flash-attention
# vi setup.py -c ':%s/c++20/c++17/g' -c ':wq'
# CC=cc PYTORCH_ROCM_ARCH='gfx90a' GPU_ARCHS='gfx90a' pip install -v .
# cd ..

# RCCL plugin
echo "Installing RCCL Plugin"
git clone --recursive --depth=1 https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl 
cd aws-ofi-rccl
libfabric_path=/opt/cray/libfabric/1.15.2.0
./autogen.sh
export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/lib:$LD_LIBRARY_PATH
CC=cc CFLAGS=-I/opt/rocm-$rocm_version/include ./configure \
    --with-libfabric=$libfabric_path --with-rccl=/opt/rocm-$rocm_version --enable-trace \
    --prefix=$PWD --with-hip=/opt/rocm-$rocm_version --with-mpi=$MPICH_DIR
make
make install
cd ..
tar -cvzf aws-ofi-rccl.tar.gz aws-ofi-rccl/
#echo "export WRKSPC_GB=/lustre/orion/$PROJ_NAME/scratch/$USER/gordon-bell" >> ~/.bashrc


