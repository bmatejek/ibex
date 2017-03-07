cimport cython
cimport numpy as np
import numpy as np
import ctypes
from math import ceil

cdef extern from "cpp-skeletonize.h":
    unsigned long *Skeletonize(unsigned long *segmentation, int zres, int yres, int xres)

def skeletonize(segmentation):
    (zres, yres, xres) = segmentation.shape

    cdef np.ndarray[unsigned long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_uint64)

    cdef unsigned long *cpp_dt = Skeletonize(&(cpp_segmentation[0,0,0]), zres, yres, xres)
    cdef unsigned long[:] tmp_cpp_dt = <unsigned long[:zres*yres*xres]> cpp_dt
    dt = np.asarray(tmp_cpp_dt)

    from matplotlib import pyplot as plt

    dt = np.reshape(dt, (zres, yres, xres), order='c')

    plt.imshow(dt[0,:,:])
    plt.savefig('test.png')

    plt.imshow(segmentation[0,:,:])
    plt.savefig('segmentation.png')

    import h5py

    with h5py.File('test.h5', 'w') as hf:
        hf.create_dataset('main', data=dt)