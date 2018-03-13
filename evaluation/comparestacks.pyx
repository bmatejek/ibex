cimport cython
cimport numpy as np
import numpy as np
import ctypes

from ibex.transforms import distance, seg2seg
from cremi_python.cremi.evaluation.voi import voi
from cremi_python.cremi.evaluation import border_mask


cdef extern from 'cpp-comparestacks.h':
    void CppEvaluate(long *segmentation, long *gold, long resolution[3], unsigned char mask_ground_truth)

def PrincetonEvaluate(segmentation, gold, dilate_ground_truth=1, mask_ground_truth=True, filtersize=0):
    # make sure these elements are the same size
    assert (segmentation.shape == gold.shape)

    # remove all small connected components
    if filtersize > 0:
        segmentation = seg2seg.RemoveSmallConnectedComponents(segmentation, filtersize)
        gold = seg2seg.RemoveSmallConnectedComponents(gold, filtersize)
    if dilate_ground_truth > 0:
        gold = distance.DilateData(gold, dilate_ground_truth)

    # convert to c++ arrays
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)

    cdef np.ndarray[long, ndim=3, mode='c'] cpp_gold
    cpp_gold = np.ascontiguousarray(gold, dtype=ctypes.c_int64)

    zres, yres, xres = segmentation.shape
    CppEvaluate(&(cpp_segmentation[0,0,0]), &(cpp_gold[0,0,0]), [zres, yres, xres], mask_ground_truth)



def CremiEvaluate(segmentation, gold, dilate_ground_truth=1, mask_ground_truth=True, mask_segmentation=False, filtersize=0):
    # make sure these elements are the same size
    assert (segmentation.shape == gold.shape)

    # remove all small connected components
    if filtersize > 0:
        segmentation = seg2seg.RemoveSmallConnectedComponents(segmentation, filtersize)
        gold = seg2seg.RemoveSmallConnectedComponents(gold, filtersize)
    if dilate_ground_truth > 0:
        masked_gold = np.zeros(gold.shape, dtype=gold.dtype)
        border_mask.create_border_mask(gold, masked_gold, dilate_ground_truth, 0)
    else:
        masked_gold = np.copy(gold)

    # run the cremi variation of information algorithm
    if mask_ground_truth: gold_mask = [0]
    else: gold_mask = []
    if mask_segmentation: segment_mask = [0]
    else: segment_mask = []

    vi_split, vi_merge = voi(segmentation, masked_gold, segment_mask, gold_mask)
    print 'Variation of Information Full: {}'.format(vi_split + vi_merge)
    print 'Variation of Information Merge: {}'.format(vi_merge)
    print 'Variation of Information Split: {}'.format(vi_split)