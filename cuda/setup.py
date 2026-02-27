"""
Build the flash_attn_cuda PyTorch extension.

Usage:
    pip install -e .          # development install
    python setup.py build_ext --inplace   # build in-place
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

# Common NVCC flags — match the CMake build.
nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--extended-lambda",
    "-std=c++17",
    # Generate code for common architectures.
    "-gencode=arch=compute_70,code=sm_70",   # V100
    "-gencode=arch=compute_80,code=sm_80",   # A100
    "-gencode=arch=compute_86,code=sm_86",   # RTX 30xx
    "-gencode=arch=compute_89,code=sm_89",   # RTX 40xx
    "-gencode=arch=compute_90,code=sm_90",   # H100
]

ext_modules = [
    CUDAExtension(
        name="flash_attn_cuda",
        sources=[
            "src/torch_bindings.cpp",
            "src/naive_attention.cu",
            "src/flash_attn_fwd.cu",
            "src/flash_attn_bwd.cu",
        ],
        include_dirs=[os.path.join(ROOT, "include")],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags,
        },
    ),
]

setup(
    name="flash_attn_cuda",
    version="0.1.0",
    description="Flash Attention 2 — from-scratch CUDA implementation",
    author="Timothee Tavernier",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch>=2.0"],
)
