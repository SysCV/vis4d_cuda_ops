"""Setup vis4d cuda ops package.

Modified from Deformable DETR: https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops
"""
import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": ["-O2"]}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-O2",
        ]
    else:
        raise NotImplementedError("Cuda is not available")

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "vis4d_cuda_ops",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="vis4d_cuda_ops",
    version="0.0",
    author="VIS @ ETH",
    author_email="i@yf.io",
    url="https://github.com/syscv/vis4d_cuda_ops",
    description="PyTorch Wrapper for CUDA Functions of Vis4D",
    packages=find_packages(exclude=("configs", "tests")),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)