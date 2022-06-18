import os, sysconfig, shutil
from setuptools import setup
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, BuildExtension

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
LIB_PATH = os.getenv('LD_LIBRARY_PATH').split(':')[-1]
PYTHON_SITE = sysconfig.get_paths()['purelib']

extension_fn = CUDAExtension if WITH_CUDA else CppExtension
devices = ['cpu', 'cuda'] if WITH_CUDA else ['cpu']
define_macros = [('WITH_CUDA', None)] if WITH_CUDA else []
libraries = [f'_segment_csr_{dev}' for dev in devices]

print('Device:', devices)

for lib in libraries:
    fsrc = os.path.join(PYTHON_SITE, f'torch_scatter/{lib}.so')
    fdst = os.path.join(LIB_PATH, f'lib{lib}.so')
    if not os.path.isfile(fdst):
        shutil.copyfile(fsrc, fdst)

setup(
    name = 'torch_sampling',
    version = '1.0.0',
    ext_modules = [extension_fn(
        name = 'torch_sampling', 
        sources = ['csrc/torch_scatter.cpp', 'csrc/tsp.cpp'],
        define_macros = define_macros,
        libraries = libraries,
        extra_compile_args = ['-fPIC', '-O3'],
    )],
    cmdclass = {
        'build_ext': BuildExtension,
    },
)
