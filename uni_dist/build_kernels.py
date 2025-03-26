
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import pathlib
import subprocess

import torch
from torch.utils import cpp_extension

# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""

BUILT=False

import mpi4py

def build():
    # Check if cuda 11 is installed for compute capability 8.0
    global BUILT 
    if BUILT:
        return
    cc_flag = []
    if torch.version.hip is None:
        _, bare_metal_major, bare_metal_minor = _get_cuda_bare_metal_version(
            cpp_extension.CUDA_HOME)
        if int(bare_metal_major) >= 11:
            cc_flag.append('-gencode')
            cc_flag.append('arch=compute_80,code=sm_80')
            if int(bare_metal_minor) >= 1:
                cc_flag.append('-gencode')
                cc_flag.append('arch=compute_86,code=sm_86')
            if int(bare_metal_minor) >= 4:
                cc_flag.append('-gencode')
                cc_flag.append('arch=compute_87,code=sm_87')
            if int(bare_metal_minor) >= 8:
                cc_flag.append('-gencode')
                cc_flag.append('arch=compute_89,code=sm_89')
        if int(bare_metal_major) >= 12:
            cc_flag.append('-gencode')
            cc_flag.append('arch=compute_90,code=sm_90')

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute() / "src"
    buildpath = pathlib.Path(__file__).parent.absolute() / 'build'
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_c_flags, extra_cuda_flags, extra_include_paths, extra_link_flags):
        if torch.version.hip is not None:
            extra_cuda_cflags=['-O3'] + extra_cuda_flags + cc_flag
        else:
            extra_cuda_cflags=['-O3',
                               '-gencode', 'arch=compute_70,code=sm_70',
                               '--use_fast_math'] + extra_cuda_flags + cc_flag

        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=extra_c_flags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
            verbose=True, #(rank==0)
            extra_ldflags=extra_link_flags
        )

    
    if torch.version.hip is not None:
        extra_include_paths=["/opt/rocm-6.2.4/include"]
    else:
        extra_include_paths=[]

    mpi4py_incl_path = mpi4py.get_include()
    extra_include_paths.append(mpi4py_incl_path)
    extra_include_paths.append(os.path.join(str(pathlib.Path(__file__).parent.absolute()), "include"))
    

    if torch.version.hip is not None:
         extra_cuda_flags = ['-D__HIP_NO_HALF_OPERATORS__=1',
                            '-D__HIP_NO_HALF_CONVERSIONS__=1'
                            '-D__HIP_ROCclr__', 
                            '-D__HIP_ARCH_GFX90A__=1',
                            '-DUSE_ROCM', 
                            '-DUSE_MPI', 
                            '-x hip',
                            ]
         extra_link_flags = ['-lamdhip64', 
                             '-lrocblas',
                             '-L/opt/rocm-6.2.4/lib']
    else:
         extra_cuda_flags = ['-U__CUDA_NO_HALF_OPERATORS__',
                            '-U__CUDA_NO_HALF_CONVERSIONS__',
                            '--expt-relaxed-constexpr',
                            '--expt-extended-lambda',
                            '-DUSE_CUDA'
                            ]
         extra_c_flags = ['-DUSE_CUDA']
         extra_link_flags = []

    extra_c_flags += ['-O3', '-std=c++17', "-I"+mpi4py.get_include()] 
    sources=[srcpath / 'yacl.cpp', 
             srcpath / 'all_gather.cpp', 
             srcpath / 'reduce_scatter.cpp',
             srcpath / 'common.cu']

   

    yacl = _cpp_extention_load_helper(
        "yacl",
        sources, 
        extra_c_flags,
        extra_cuda_flags, 
        extra_include_paths, 
        extra_link_flags)
    BUILT = True

def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
