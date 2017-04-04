cimport cython
cimport numpy as np
import ctypes
import numpy as np
import h5py
import struct
import os

cdef extern from 'cpp-seg2seg.h':
    unsigned long *CppMapLabels(unsigned long *segmentation, unsigned long *mapping, unsigned long nentries)


# map the labels from this segmentation
def MapLabels(segmentation, mapping):
    # get the size of the data
    zres, yres, xres = segmentation.shape
    nentries = segmentation.size

    cdef np.ndarray[unsigned long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_uint64)
    cdef np.ndarray[unsigned long, ndim=1, mode='c'] cpp_mapping
    cpp_mapping = np.ascontiguousarray(mapping, dtype=ctypes.c_uint64)

    cdef unsigned long *mapped_segmentation = CppMapLabels(&(cpp_segmentation[0,0,0]), &(cpp_mapping[0]), nentries)
    cdef unsigned long[:] tmp_segmentation = <unsigned long[:segmentation.size]> mapped_segmentation

    return np.reshape(np.asarray(tmp_segmentation), (zres, yres, xres))

# remove the components less than min size
def RemoveSmallConnectedComponents(segmentation, min_size=64):
    original_dtype = segmentation.dtype

    # slight shortcut
    if min_size == 0: return segmentation

    # get the number of bins for each segment
    component_sizes = np.bincount(segmentation.ravel())

    # find the components less than a certain size
    small_components = component_sizes < min_size

    # find the locations where these components exist
    small_locations = small_components[segmentation]

    # set the values to 0
    segmentation[small_locations] = 0

    return segmentation.astype(original_dtype)

def ReduceLabels(segmentation):
    # get the unique labels
    unique = np.unique(segmentation)

    # get the maximum label for the segment
    maximum_label = np.amax(segmentation) + 1

    # create an array from original segment id to reduced id
    mapping = np.zeros(maximum_label, dtype=np.int64) - 1

    for ie, label in enumerate(unique):
        mapping[label] = ie

    # return the forward and reverse mapping
    return mapping, unique

# function to split segmentation into two parts (along the z dimension)
def SplitSegmentation(filename, dataset, axis=0, threshold=0.5):
    # get file components
    components = os.path.split(filename)
    folder = components[0]
    prefix = components[1].split('_')[0]
    suffix = components[1].split('_')[1]

    # open data
    with h5py.File(filename, 'r') as hf:
        data = np.array(hf[dataset])

    separation = int(threshold * data.shape[axis])

    ## TODO ONLY WORKS FOR Z DIMENSION
    training_data = data[0:separation,:,:]
    validation_data = data[separation:,:,:]

    # free data memory
    del data

    # create the training data
    training_filename = folder + '/train_' + prefix + '_' + suffix
    validation_filename = folder + '/validation_' + prefix + '_' + suffix

    with h5py.File(training_filename, 'w') as hf:
        hf.create_dataset(dataset, data=training_data)
    with h5py.File(validation_filename, 'w') as hf:
        hf.create_dataset(dataset, data=validation_data)