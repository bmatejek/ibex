import struct
import numpy as np
from numba import jit
from ibex.utilities import dataIO
from ibex.utilities.constants import *
from ibex.geometry import ibbox
from ibex.transforms import seg2seg


# go from world coordinates to grid coordinates
def WorldToGrid(world_position, bounding_box):
    zdiff = world_position[IB_Z] - bounding_box.Min(IB_Z)
    ydiff = world_position[IB_Y] - bounding_box.Min(IB_Y)
    xdiff = world_position[IB_X] - bounding_box.Min(IB_X)

    return (zdiff, ydiff, xdiff)



# find the overlap candidates
@jit(nopython=True)
def FindOverlapCandidates(segmentation_one, segmentation_two, candidates):
    assert (segmentation_one.shape == segmentation_two.shape)

    # get the dimensions of the shapes
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



# save the candidates
def SaveFeatures(prefix_one, prefix_two, candidates, threshold):
    # get the output filename
    filename = 'features/ebro/{}-{}-{}.candidates'.format(prefix_one, prefix_two, threshold)
    with open(filename, 'wb') as fd:
        fd.write(struct.pack('Q', len(candidates) - 1))

        # save the candidates
        for candidate in candidates:
            if not candidate[0] or not candidate[1]: continue

            fd.write(struct.pack('QQ', candidate[0], candidate[1]))



# generate the candidates given two segmentations
def GenerateFeatures(prefix_one, prefix_two, threshold=10000):
    # read the meta data for both prefixes
    meta_data_one = dataIO.ReadMetaData(prefix_one)
    meta_data_two = dataIO.ReadMetaData(prefix_two)

    # get the bounding boxes for both 
    bbox_one = meta_data_one.BBox()
    bbox_two = meta_data_two.BBox()

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
    segment_one = segmentation_one[mins_one[IB_Z]:maxs_one[IB_Z],mins_one[IB_Y]:maxs_one[IB_Y],mins_one[IB_X]:maxs_one[IB_X]]
    segment_two = segmentation_two[mins_two[IB_Z]:maxs_two[IB_Z],mins_two[IB_Y]:maxs_two[IB_Y],mins_two[IB_X]:maxs_two[IB_X]]

      # create an empty set and add dumby variable to 
    candidates = set()
    candidates.add((np.uint64(0), np.uint64(0)))
    FindOverlapCandidates(segment_one, segment_two, candidates)

    # save the features
    SaveFeatures(prefix_one, prefix_two, candidates, threshold)