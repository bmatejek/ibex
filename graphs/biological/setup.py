from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

home_dir = os.path.expanduser('~')

extensions = [
    Extension(
        name='node_generation',
        include_dirs=[np.get_include()],
        sources=['node_generation.pyx', 'cpp-node_generation.cpp'],
        extra_compile_args=['-O4', '-std=c++11'],
        language='c++'
    )
]

setup(
    name='node_generation',
    ext_modules = cythonize(extensions)
)
