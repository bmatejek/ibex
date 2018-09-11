cimport cython
cimport numpy as np

import ctypes
import numpy as np


cdef extern from 'cpp-distance.h':
    float *CppTwoDimensionalDistanceTransform(long *data, long grid_size[3])
    void CppDilateData(long *data, long grid_size[3], float distance)


# get the two dimensional distance transform
def TwoDimensionalDistanceTransform(data):
    # convert numpy array to c++
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_data = np.ascontiguousarray(data, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_input_grid_size = np.ascontiguousarray(data.shape, dtype=ctypes.c_int64)

    # get the distance transform
    cdef float[:] distances = <float[:data.size]>CppTwoDimensionalDistanceTransform(&(cpp_data[0,0,0]), &(cpp_input_grid_size[0]))

    del cpp_input_grid_size

    return np.reshape(np.asarray(distances), data.shape)



# dilate segments from boundaries
def DilateData(data, distance):
    # convert numpy array to c++
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_data = np.ascontiguousarray(data, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_input_grid_size = np.ascontiguousarray(data.shape, dtype=ctypes.c_int64)

    # dilate the data by distance
    CppDilateData(&(cpp_data[0,0,0]), &(cpp_input_grid_size[0]), float(distance))

    del cpp_input_grid_size