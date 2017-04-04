from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'seg2gold',
        include_dirs=[np.get_include()],
        sources=['seg2gold.pyx', 'cpp-seg2gold.cpp'],
        extra_compile_args=['-O4', '-std=c++0x'],
        language='c++'
    )
]

setup(
    ext_modules = cythonize(extensions)
)