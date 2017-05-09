from ibex.transforms import seg2seg
from ibex.data_structures import UnionFind
from ibex.utilities import dataIO
import numpy as np
import os
import struct



def Agglomerate(prefix, model_prefix, threshold=0.5):
    # read the segmentation data 
    segmentation = dataIO.ReadSegmentationData(prefix)

    # get the multicut filename (with graph weights)
    multicut_filename = 'multicut/{}-{}.graph'.format(model_prefix, prefix)

    # get the maximum segmentation value
    max_value = np.uint64(np.amax(segmentation) + 1)

    # create union find data structure
    union_find = [UnionFind.UnionFindElement(iv) for iv in range(max_value)]

    # read in all of the labels and merge the result
    with open(multicut_filename, 'rb') as fd:
        # read the number of vertices and edges
        nvertices, nedges, = struct.unpack('QQ', fd.read(16))

        # read in all of the edges
        for ie in range(nedges):
            # read in both labels
            label_one, label_two, = struct.unpack('QQ', fd.read(16))

            # skip over the reduced labels
            fd.read(16)

            # read in the edge weight
            edge_weight, = struct.unpack('d', fd.read(8))

            # merge label one and label two in the union find data structure
            if (edge_weight > threshold):
                UnionFind.Union(union_find[label_one], union_find[label_two])

    # create a mapping
    mapping = np.zeros(max_value, dtype=np.uint64)

    # update the segmentation
    for iv in range(max_value):
        label = UnionFind.Find(union_find[iv]).label

        mapping[iv] = label

    # update the labels
    agglomerated_segmentation = seg2seg.MapLabels(segmentation, mapping)

    gold_filename = 'gold/{}_gold.h5'.format(prefix)

    # TODO fix this code temporary filename
    agglomeration_filename = 'multicut/{}-agglomerate.h5'.format(prefix)

    # temporary - write h5 file
    dataIO.WriteH5File(agglomerated_segmentation, agglomeration_filename, 'stack')

    import time
    start_time = time.time()
    print 'Agglomeration: '
    # create the command line 
    command = '~/software/PixelPred2Seg/comparestacks --stack1 {} --stackbase {} --dilate1 1 --dilatebase 1 --relabel1 --relabelbase --filtersize 100 --anisotropic'.format(agglomeration_filename, gold_filename)

    # execute the command
    os.system(command)
    print time.time() - start_time

