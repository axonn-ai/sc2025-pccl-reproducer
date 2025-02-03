This was before I discovered the following NIC mapping policy.

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY="GPU"
export MPICH_OFI_NUM_NICS=4
export MPICH_OFI_VERBOSE=1