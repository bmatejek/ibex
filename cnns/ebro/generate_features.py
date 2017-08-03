import struct
import numpy as np
from numba import jit

from ibex.geometry import ib3shapes
from ibex.transforms import seg2seg
from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.cnns.ebro.util import WorldToGrid, ReadFeatures



# find all overlap candidates for these segmentations
@jit(nopython=True)
def FindOverlapCandidates(segmentation_one, segmentation_two, candidates):
    assert (segmentation_one.shape == segmentation_two.shape)
    zres, yres, xres = segmentation_one.shape

    # iterate over all voxels
    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if not segmentation_one[iz,iy,ix]: continue
                if not segmentation_two[iz,iy,ix]: continue

                candidates.add((segmentation_one[iz,iy,ix], segmentation_two[iz,iy,ix]))



# find the centers for all overlapping candidates
@jit(nopython=True)
def FindCenters(segmentation_one, segmentation_two, forward_mapping_one, forward_mapping_two, sums, counter):
    assert (segmentation_one.shape == segmentation_two.shape)
    zres, yres, xres = segmentation_one.shape

    # iterate over all voxels
    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                # get the forward mapping index
                index_one = forward_mapping_one[segmentation_one[iz,iy,ix]]
                index_two = forward_mapping_two[segmentation_two[iz,iy,ix]]

                # increment counters
                sums[index_one,index_two,IB_Z] += iz
                sums[index_one,index_two,IB_Y] += iy
                sums[index_one,index_two,IB_X] += ix
                counter[index_one,index_two] += 1



@jit(nopython=True)
def SearchCandidate(ones, twos, labels, overlap, mins, maxs):
    assert (ones.shape == twos.shape)
    zres, yres, xres = ones.shape

    # get convenient booleans
    minx = (mins[IB_X] < 0)
    maxx = (maxs[IB_X] >= xres)
    miny = (mins[IB_Y] < 0)
    maxy = (maxs[IB_Y] >= yres)
    minz = (mins[IB_Z] < 0)
    maxz = (maxs[IB_Z] >= zres)

    # check z and y slices 
    if not overlap == IB_X:
        for iz in range(zres):
            for iy in range(yres):
                if minx and (ones[iz,iy,0] == labels[0] or twos[iz,iy,0] == labels[1]): return False
                if maxx and (ones[iz,iy,xres-1] == labels[0] or twos[iz,iy,xres-1] == labels[1]): return False

    # check z and x slices
    if not overlap == IB_Y:
        for iz in range(zres):
            for ix in range(xres):
                if miny and (ones[iz,0,ix] == labels[0] or twos[iz,0,ix] == labels[1]): return False
                if maxy and (ones[iz,yres-1,ix] == labels[0] or twos[iz,yres-1,ix] == labels[1]): return False

    if not overlap == IB_Z:
        for iy in range(yres):
            for ix in range(xres):
                if minz and (ones[0,iy,ix] == labels[0] or twos[0,iy,ix] == labels[1]): return False
                if maxz and (ones[zres-1,iy,ix] == labels[0] or twos[zres-1,iy,ix] == labels[1]): return False

    return True



# prune the candidates so they do not overflow on boundary
def PruneCandidates(segmentation_one, segmentation_two, candidates, centers, radii, overlap):
    assert (segmentation_one.shape == segmentation_two.shape)
    zres, yres, xres = segmentation_one.shape

    # get the global grid box
    global_box = ib3shapes.IBBox((0, 0, 0), (zres, yres, xres))

    indices = []
    # iterate over every candidate
    for iv, candidate in enumerate(candidates):
        # create a local bounding box for this feature
        cz, cy, cx = centers[iv]
        mins = (cz - radii[IB_Z], cy - radii[IB_Y], cx - radii[IB_X])
        maxs = (cz + radii[IB_Z], cy + radii[IB_Y], cx + radii[IB_X])
        
        # first check if the local box is contained
        if 0 < mins[IB_X] and 0 < mins[IB_Y] and 0 < mins[IB_Z] and maxs[IB_X] < xres and maxs[IB_Y] < yres and maxs[IB_Z] < zres: indices.append(iv)
        # also allow otherwise valid candidates
        elif SearchCandidate(segmentation_one, segmentation_two, candidate, overlap, mins, maxs): indices.append(iv)

    # return which indices are valid
    return indices



def SaveFeatures(prefix_one, prefix_two, candidates, centers, counter, threshold, maximum_distance):
    # get output filename
    filename = 'features/ebro/{}-{}-{}-{}nm.candidates'.format(prefix_one, prefix_two, threshold, maximum_distance)

    # write all of the features
    ncandidates = len(candidates)
    with open(filename, 'wb') as fd:
        fd.write(struct.pack('i', ncandidates))
        for iv in range(ncandidates):
            fd.write(struct.pack('QQQQQ', candidates[iv][0], candidates[iv][1], centers[iv][IB_Z], centers[iv][IB_Y], centers[iv][IB_X]))

    filename = 'features/ebro/{}-{}-{}-{}nm.counts'.format(prefix_one, prefix_two, threshold, maximum_distance)

    # write all of the counts
    with open(filename, 'wb') as fd:
        fd.write(struct.pack('i', ncandidates))
        for iv in range(ncandidates):
            fd.write(struct.pack('QQQ', counter[iv][0], counter[iv][1], counter[iv][2]))



# generate the candidates given two segmentations
def GenerateFeatures(prefix_one, prefix_two, threshold, maximum_distance):
    # read in all relevant information
    segmentation_one = dataIO.ReadSegmentationData(prefix_one)
    segmentation_two = dataIO.ReadSegmentationData(prefix_two)
    assert (segmentation_one.shape == segmentation_two.shape)
    bbox_one = dataIO.GetWorldBBox(prefix_one)
    bbox_two = dataIO.GetWorldBBox(prefix_two)
    world_res = dataIO.Resolution(prefix_one)
    assert (world_res == dataIO.Resolution(prefix_two))

    # get the radii for the relevant region
    radii = (int(maximum_distance / world_res[IB_Z] + 0.5), int(maximum_distance / world_res[IB_Y] + 0.5), int(maximum_distance / world_res[IB_X] + 0.5))



    # parse out small segments
    segmentation_one = seg2seg.RemoveSmallConnectedComponents(segmentation_one, min_size=threshold)
    segmentation_two = seg2seg.RemoveSmallConnectedComponents(segmentation_two, min_size=threshold)

    # get the bounding box for the intersection
    world_box = ib3shapes.IBBox(bbox_one.mins, bbox_one.maxs)
    world_box.Intersection(bbox_two)

    # get the mins and maxs of truncated box
    mins_one = WorldToGrid(world_box.mins, bbox_one)
    mins_two = WorldToGrid(world_box.mins, bbox_two)
    maxs_one = WorldToGrid(world_box.maxs, bbox_one)
    maxs_two = WorldToGrid(world_box.maxs, bbox_two)

    # get the relevant subsections
    segmentation_one = segmentation_one[mins_one[IB_Z]:maxs_one[IB_Z], mins_one[IB_Y]:maxs_one[IB_Y], mins_one[IB_X]:maxs_one[IB_X]]
    segmentation_two = segmentation_two[mins_two[IB_Z]:maxs_two[IB_Z], mins_two[IB_Y]:maxs_two[IB_Y], mins_two[IB_X]:maxs_two[IB_X]]



    # create an emptu set and add dumby variable for numba
    candidates_set = set()
    # this set represents tuples of labels from GRID_ONE and GRID_TWO
    candidates_set.add((np.uint64(0), np.uint64(0)))
    FindOverlapCandidates(segmentation_one, segmentation_two, candidates_set)

    # get the reverse mappings
    forward_mapping_one, reverse_mapping_one = seg2seg.ReduceLabels(segmentation_one)
    forward_mapping_two, reverse_mapping_two = seg2seg.ReduceLabels(segmentation_two)



    # get the number of unique labels
    nlabels_one = reverse_mapping_one.size
    nlabels_two = reverse_mapping_two.size

    # calculate the center of overlap regions
    sums = np.zeros((nlabels_one, nlabels_two, 3), dtype=np.uint64)
    counter = np.zeros((nlabels_one, nlabels_two), dtype=np.uint64)
    FindCenters(segmentation_one, segmentation_two, forward_mapping_one, forward_mapping_two, sums, counter)

    # get the number of occurrences of all labels
    _, counts_one = np.unique(segmentation_one, return_counts=True)
    _, counts_two = np.unique(segmentation_two, return_counts=True)



    # iterate through candidate and locate centers
    candidates = []
    centers = []
    counts = []
    for candidate in candidates_set:
        # skip extracellular space
        if not candidate[0] or not candidate[1]: continue

        # get forward mapping
        index_one = forward_mapping_one[candidate[0]]
        index_two = forward_mapping_two[candidate[1]]

        count = counter[index_one][index_two]
        center = (int(sums[index_one, index_two, IB_Z] / count + 0.5), int(sums[index_one, index_two, IB_Y] / count + 0.5), int(sums[index_one, index_two, IB_X] / count + 0.5))

        # append to the lists
        candidates.append(candidate)
        centers.append(center)
        counts.append((counts_one[index_one], counts_two[index_two], count))



    # find which dimension causes overlap
    if not bbox_one.mins[IB_X] == bbox_two.mins[IB_X]: overlap = IB_X
    if not bbox_one.mins[IB_Y] == bbox_two.mins[IB_Y]: overlap = IB_Y
    if not bbox_one.mins[IB_Z] == bbox_two.mins[IB_Z]: overlap = IB_Z



    # prune the candidates
    indices = PruneCandidates(segmentation_one, segmentation_two, candidates, centers, radii, overlap)
    pruned_candidates = []
    pruned_centers = []
    pruned_counts = []
    for index in indices:
        # add the candidates
        pruned_candidates.append(candidates[index])
        pruned_counts.append(counts[index])    

        center = (centers[index][IB_Z] + world_box.mins[IB_Z], centers[index][IB_Y] + world_box.mins[IB_Y], centers[index][IB_X] + world_box.mins[IB_X])
        pruned_centers.append(center)

    # save all features
    SaveFeatures(prefix_one, prefix_two, pruned_candidates, pruned_centers, pruned_counts, threshold, maximum_distance)
