from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='_C',
    ext_modules=[
        CUDAExtension(
            name='_C',
            sources=['csrc/connected_components.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__'
                ],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)