import struct
import numpy as np
from numba import jit
from ibex.utilities import dataIO
from ibex.utilities.constants import *
from ibex.geometry import ibbox
from ibex.transforms import seg2seg
from util import WorldToGrid
from scipy import sparse



# find the overlap candidates
@jit(nopython=True)
def FindOverlapCandidates(segmentation_one, segmentation_two, candidates):
    assert (segmentation_one.shape == segmentation_two.shape)

    # get the dimensions of the datasets
    zdim, ydim, xdim = segmentation_one.shape

    # iterate over every voxel
    for iz in range(zdim):
        for iy in range(ydim):
            for ix in range(xdim):
                # get the label from each segment
                label_one = segmentation_one[iz,iy,ix]
                label_two = segmentation_two[iz,iy,ix]

                # skip extra cellular space
                if not label_one: continue
                if not label_two: continue

                # add this label pair to the set
                candidates.add((label_one, label_two))



@jit(nopython=True)
def FindCenters(segmentation_one, forward_mapping_one, segmentation_two, forward_mapping_two, xsums, ysums, zsums, counter):
    assert (segmentation_one.shape == segmentation_two.shape)

    # get the dimensions of the datasets
    zdim, ydim, xdim = segmentation_one.shape

    # iterate over every voxel
    for iz in range(zdim):
        for iy in range(ydim):
            for ix in range(xdim):
                # get the indices in the sum arrays
                index_one = forward_mapping_one[segmentation_one[iz,iy,ix]]
                index_two = forward_mapping_two[segmentation_two[iz,iy,ix]]

                # add to the sum arrays
                xsums[index_one,index_two] += ix
                ysums[index_one,index_two] += iy
                zsums[index_one,index_two] += iz
                counter[index_one,index_two] += 1



@jit(nopython=True)
def PruneCandidate(segmentation_one, segmentation_two, candidate, overlap_dimension, mins, maxs):
    # get the dimensions of the datasets
    zdim, ydim, xdim = segmentation_one.shape

    # otherwise see if this presents an issue
    if overlap_dimension == IB_X:
        for ix in range(xdim):
            for iz in range(zdim):
                if (mins[IB_Y] < 0):
                    if segmentation_one[iz,0,ix] == candidate[0] or segmentation_two[iz,0,ix] == candidate[1]:
                        return False
                if (maxs[IB_Y] >= ydim):
                    if segmentation_one[iz,ydim-1,ix] == candidate[0] or segmentation_two[iz,ydim-1,ix] == candidate[1]:
                        return False
                                            
            for iy in range(ydim):
                if (mins[IB_Z] < 0):
                    if segmentation_one[0,iy,ix] == candidate[0] or segmentation_two[0,iy,ix] == candidate[1]:
                        return False
                if (maxs[IB_Z] >= zdim):
                    if segmentation_one[zdim-1,iy,ix] == candidate[0] or segmentation_two[zdim-1,iy,ix] == candidate[1]:
                        return False
    elif overlap_dimension == IB_Y:
        for iy in range(ydim):
            for iz in range(zdim):
                if (mins[IB_X] < 0):
                    if segmentation_one[iz,iy,0] == candidate[0] or segmentation_two[iz,iy,0] == candidate[1]:
                        return False
                if (maxs[IB_X] >= xdim):
                    if segmentation_one[iz,iy,xdim-1] == candidate[0] or segmentation_two[iz,iy,xdim-1] == candidate[1]:
                        return False
            for ix in range(xdim):
                if (mins[IB_Z] < 0):
                    if segmentation_one[0,iy,ix] == candidate[0] or segmentation_two[0,iy,ix] == candidate[1]:
                        return False
                if (maxs[IB_Z] >= zdim):
                    if segmentation_one[zdim-1,iy,ix] == candidate[0] or segmentation_two[zdim-1,iy,ix] == candidate[1]:
                        return False
    elif overlap_dimension == IB_Z:
        for iz in range(zdim):
            for ix in range(xdim):
                if (mins[IB_Y] < 0):
                    if segmentation_one[iz,0,ix] == candidate[0] or segmentation_two[iz,0,ix] == candidate[1]:
                        return False
                if (maxs[IB_Y] >= ydim):
                    if segmentation_one[iz,ydim-1,ix] == candidate[0] or segmentation_two[iz,ydim-1,ix] == candidate[1]:
                        return False
            for iy in range(ydim):
                if (mins[IB_X] < 0):
                    if segmentation_one[iz,iy,0] == candidate[0] or segmentation_two[iz,iy,0] == candidate[1]:
                        return False
                if (maxs[IB_X] >= xdim):
                    if segmentation_one[iz,iy,xdim-1] == candidate[0] or segmentation_two[iz,iy,xdim-1] == candidate[1]:
                        return False
    return True



def PruneCandidates(segmentation_one, segmentation_two, candidates, centers, counts, radii, overlap_dimension):
    assert (segmentation_one.shape == segmentation_two.shape)
    
    # create arrays for reduced candidates
    pruned_candidates = []
    pruned_centers = []
    pruned_counts = []

    # get the dimensions of the grid
    zdim, ydim, xdim = segmentation_one.shape
    
    for iv, candidate in enumerate(candidates):
        cz, cy, cx = centers[iv]
        
        # create a bounding box for this feature
        mins = (cz - radii[IB_Z], cy - radii[IB_Y], cx - radii[IB_X])
        maxs = (cz + radii[IB_Z], cy + radii[IB_Y], cx + radii[IB_X])
        
        # if completely contained, fine
        if (0 <= mins[IB_X]) and (0 <= mins[IB_Y]) and (0 <= mins[IB_Z]) and (maxs[IB_X] < xdim) and (maxs[IB_Y] < ydim) and (maxs[IB_Z] < zdim):
            pruned_candidates.append(candidate)
            pruned_centers.append(centers[iv])
            pruned_counts.append(counts[iv])
            continue
        # otherwise make sure the candidate is valid
        if PruneCandidate(segmentation_one, segmentation_two, candidate, overlap_dimension, mins, maxs):
            pruned_candidates.append(candidate)
            pruned_centers.append(centers[iv])
            pruned_counts.append(counts[iv])
            
    return pruned_candidates, pruned_centers, pruned_counts
                


# save the candidates
def SaveFeatures(prefix_one, prefix_two, candidates, centers, counts, bbox, threshold, radius):
    # get the output filename
    filename = 'features/ebro/{}-{}-{}-{}nm.candidates'.format(prefix_one, prefix_two, threshold, radius)

    # iterate through every candidate
    with open(filename, 'wb') as fd:
        fd.write(struct.pack('Q', len(candidates)))

        # save the candidates
        for iv, candidate in enumerate(candidates):
            fd.write(struct.pack('QQQQQ', candidate[0], candidate[1], centers[iv][IB_X] + bbox.XMin(), centers[iv][IB_Y] + bbox.YMin(), centers[iv][IB_Z] + bbox.ZMin()))

    # get the output filename
    filename = 'features/ebro/{}-{}-{}-{}nm.counts'.format(prefix_one, prefix_two, threshold, radius)

    # iterate through every count
    with open(filename, 'wb') as fd:
        fd.write(struct.pack('Q', len(counts)))

        for count in counts:
            count_one, count_two, overlap_count = count
            fd.write(struct.pack('QQQd', count_one, count_two, overlap_count, float(overlap_count) / float(count_one + count_two - overlap_count)))



# generate the candidates given two segmentations
def GenerateFeatures(prefix_one, prefix_two, threshold=10000, radius=600):
    # read the meta data for both prefixes
    meta_data_one = dataIO.ReadMetaData(prefix_one)
    meta_data_two = dataIO.ReadMetaData(prefix_two)

    # get the bounding boxes for both 
    bbox_one = meta_data_one.WorldBBox()
    bbox_two = meta_data_two.WorldBBox()

    # find the intersection between the two boxes
    intersected_box = ibbox.IBBox(bbox_one.Mins(), bbox_one.Maxs())
    intersected_box.Intersection(bbox_two)

    # get the segmentation for both datasets
    segmentation_one = dataIO.ReadSegmentationData(prefix_one)
    segmentation_two = dataIO.ReadSegmentationData(prefix_two)

    # remove small components
    segmentation_one = seg2seg.RemoveSmallConnectedComponents(segmentation_one, min_size=threshold)
    segmentation_two = seg2seg.RemoveSmallConnectedComponents(segmentation_two, min_size=threshold)

    # get the min and max points for both datasets
    mins_one = WorldToGrid(intersected_box.Mins(), bbox_one)
    maxs_one = WorldToGrid(intersected_box.Maxs(), bbox_one)
    mins_two = WorldToGrid(intersected_box.Mins(), bbox_two)
    maxs_two = WorldToGrid(intersected_box.Maxs(), bbox_two)

    # get the relevant subsections
    segmentation_one = segmentation_one[mins_one[IB_Z]:maxs_one[IB_Z],mins_one[IB_Y]:maxs_one[IB_Y],mins_one[IB_X]:maxs_one[IB_X]]
    segmentation_two = segmentation_two[mins_two[IB_Z]:maxs_two[IB_Z],mins_two[IB_Y]:maxs_two[IB_Y],mins_two[IB_X]:maxs_two[IB_X]]

    # create an empty set and add dumby variable to 
    candidate_set = set()
    candidate_set.add((np.uint64(0), np.uint64(0)))
    # create four sparse matrices
    FindOverlapCandidates(segmentation_one, segmentation_two, candidate_set)
    
    # find the forward and reverse mapping for segmentations
    forward_mapping_one, reverse_mapping_one = seg2seg.ReduceLabels(segmentation_one)
    forward_mapping_two, reverse_mapping_two = seg2seg.ReduceLabels(segmentation_two)

    # get the number of unique labels
    nlabels_one = reverse_mapping_one.size
    nlabels_two = reverse_mapping_two.size

    # find the center of this feature
    xsums = np.zeros((nlabels_one, nlabels_two), dtype=np.uint64)
    ysums = np.zeros((nlabels_one, nlabels_two), dtype=np.uint64)
    zsums = np.zeros((nlabels_one, nlabels_two), dtype=np.uint64)
    counter = np.zeros((nlabels_one, nlabels_two), dtype=np.uint64)
    FindCenters(segmentation_one.astype(np.uint64), forward_mapping_one, segmentation_two, forward_mapping_two, xsums, ysums, zsums, counter)

    # get the number of occurrences for all voxels
    _, one_counts = np.unique(segmentation_one, return_counts=True)
    _, two_counts = np.unique(segmentation_two, return_counts=True)

    # iterate through all candidates and locate centers
    centers = []
    counts = []
    candidates = []
    for candidate in candidate_set:
        # skip extraceullar space
        if not candidate[0] or not candidate[1]: continue

        # get this index
        index_one = forward_mapping_one[candidate[0]]
        index_two = forward_mapping_two[candidate[1]]

        # get the center location
        xcenter = int(xsums[index_one,index_two] / counter[index_one,index_two]) 
        ycenter = int(ysums[index_one,index_two] / counter[index_one,index_two]) 
        zcenter = int(zsums[index_one,index_two] / counter[index_one,index_two])

        # add to the list of candidates
        candidates.append(candidate)

        # add to the list of centers
        centers.append((zcenter, ycenter, xcenter))

        # get the counts for this section
        counts.append((one_counts[index_one], two_counts[index_two], counter[index_one,index_two]))

    # find the overlap dimensions
    if (not bbox_one.XMin() == bbox_two.XMin()): overlap_dimension = IB_X
    elif (not bbox_one.YMin() == bbox_two.YMin()): overlap_dimension = IB_Y
    elif (not bbox_one.ZMin() == bbox_two.ZMin()): overlap_dimension = IB_Z
        
    # convert the radius into voxels
    resolution = meta_data_one.Resolution()
    radii = (int(radius / resolution[IB_Z]), int(radius / resolution[IB_Y]), int(radius / resolution[IB_X]))
        
    # prune the candidates that intersect the bounding box
    candidates, centers, counts = PruneCandidates(segmentation_one, segmentation_two, candidates, centers, counts, radii, overlap_dimension)
    
    # save the features
    SaveFeatures(prefix_one, prefix_two, candidates, centers, counts, intersected_box, threshold, radius)