from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
            'skeletonize',
            include_dirs=[np.get_include()],
            sources=['skeletonize.pyx', 'cpp-skeletonize.cpp'],
            extra_compile_args=['-O4', '-std=c++0x'],
            language='c++'
        )
]

setup(
    ext_modules = cythonize(extensions)
)
