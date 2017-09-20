import os
import h5py
import numpy as np
from ibex.data_structures import meta_data, swc

def GetWorldBBox(prefix):
    # return the bounding box for this segment
    return meta_data.MetaData(prefix).WorldBBox()



def ReadMetaData(prefix):
    # return the meta data for this prefix
    return meta_data.MetaData(prefix)



def Resolution(prefix):
    # return the resolution for this prefix
    return meta_data.MetaData(prefix).Resolution()



def ReadH5File(filename, dataset=None):
    # read the h5py file
    with h5py.File(filename, 'r') as hf:
        # read the first dataset if none given
        if dataset == None: data = np.array(hf[hf.keys()[0]])
        else: data = np.array(hf[dataset])

    # return the data
    return data



def WriteH5File(data, filename, dataset):
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset(dataset, data=data)



def ReadSegmentationData(prefix):
    filename, dataset = meta_data.MetaData(prefix).SegmentationFilename()

    return ReadH5File(filename, dataset)



def ReadGoldData(prefix):
    filename, dataset = meta_data.MetaData(prefix).GoldFilename()

    return ReadH5File(filename, dataset)



def ReadImageData(prefix):
    filename, dataset = meta_data.MetaData(prefix).ImageFilename()

    return ReadH5File(filename, dataset)



def ReadSkeletons(prefix, data):
    # read in all of the skeletons
    skeletons = []
    joints = []
    endpoints = []

    for label in np.unique(data):
        skeleton_filename = 'skeletons/{}/tree_{}.swc'.format(prefix, label)

        # see if the skeleton exists
        if not os.path.isfile(skeleton_filename):
            continue

        # read the skeleton
        skeleton = swc.Skeleton(prefix, label)

        skeletons.append(skeleton)

        # add all joints for this skeleton
        for ij in range(skeleton.NJoints()):
            joints.append(skeleton.Joint(ij))
        # add all endpoints for this skeleton
        for ip in range(skeleton.NEndPoints()):
            endpoints.append(skeleton.EndPoint(ip))

    # return all of the skeletons
    return skeletons, joints, endpoints
