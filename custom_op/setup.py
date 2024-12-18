from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_op',
    ext_modules=[
        CUDAExtension(
            name='custom_op',
            sources=['custom_op.cu'],
            extra_compile_args={
                'nvcc': ['-g']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)