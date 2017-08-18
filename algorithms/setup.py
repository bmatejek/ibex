from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

extensions = [
    Extension(
        name='multicut',
        include_dirs=[np.get_include(), '/home/bmatejek/software/graph/include', '/opt/gurobi702/linux64/include'],
        #library_dirs=['/opt/gurobi702/linux64/lib'],
        #libraries=['gurobi70', 'gurobi_c++'],
        sources=['multicut.pyx', 'cpp-multicut.cpp'],
        extra_compile_args=['-O4', '-std=c++0x'],
        language='c++'
    )
]

setup(
    name='algorithms',
    ext_modules = cythonize(extensions)
)
