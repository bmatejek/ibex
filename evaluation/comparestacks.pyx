cimport cython
cimport numpy as np
import numpy as np
import ctypes

from ibex.transforms import distance, seg2seg
from cremi_python.cremi.evaluation.voi import voi
from cremi_python.cremi.evaluation import border_mask
import scipy.sparse as sparse


cdef extern from 'cpp-comparestacks.h':
    void CppEvaluate(long *segmentation, long *gold, long resolution[3], unsigned char mask_ground_truth)


def adapted_rand(seg, gt, all_stats=False, dilate_ground_truth=1, filtersize=0):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index 
    (excluding the zero component of the original labels). Adapted 
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """

    # remove all small connected components
    if filtersize > 0:
        seg = seg2seg.RemoveSmallConnectedComponents(seg, filtersize)
        gt = seg2seg.RemoveSmallConnectedComponents(gt, filtersize)
    if dilate_ground_truth > 0:
        gt = distance.DilateData(gt, dilate_ground_truth)


    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A,:]
    b = p_ij[1:n_labels_A,1:n_labels_B]
    c = p_ij[1:n_labels_A,0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / sumB
    recall = sumAB / sumA

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    if all_stats:
       return (are, precision, recall)
    else:
       return are

def PrincetonEvaluate(segmentation, gold, dilate_ground_truth=1, mask_ground_truth=True, filtersize=0):
    # make sure these elements are the same size
    assert (segmentation.shape == gold.shape)
    assert (np.amin(segmentation) >= 0 and np.amin(gold) >= 0)

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
    assert (np.amin(segmentation) >= 0 and np.amin(gold) >= 0)

    # remove all small connected components
    if filtersize > 0:
        segmentation = seg2seg.RemoveSmallConnectedComponents(segmentation, filtersize)
        gold = seg2seg.RemoveSmallConnectedComponents(gold, filtersize)
    if dilate_ground_truth > 0:
        gold = distance.DilateData(gold, 1)

    vi_split, vi_merge = voi(segmentation, gold)
    print 'Variation of Information Full: {}'.format(vi_split + vi_merge)
    print 'Variation of Information Merge: {}'.format(vi_merge)
    print 'Variation of Information Split: {}'.format(vi_split)