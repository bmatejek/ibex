import os
import h5py
import numpy as np
from ibex.data_structures import swc

def ReadMetaData(prefix):
    # generate the meta data default filename
    filename = 'meta_data/' + prefix + '.meta'

    # open the meta data filename
    with open(filename, 'r') as fd:
        meta_data = fd.readlines()
        resolutions = meta_data[1].split('x')

    # return the resolution in nanometers (z, y, x)
    return (int(resolutions[2]), int(resolutions[1]), int(resolutions[0]))



def ReadH5File(filename, dataset):
    # read the h5py file
    with h5py.File(filename, 'r') as hf:
        data = np.array(hf[dataset])

    # return the data
    return data


def WriteH5File(data, filename, dataset):
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset(dataset, data=data)



def ReadSegmentationData(prefix):
    filename = 'rhoana/{0}_rhoana.h5'.format(prefix)

    return ReadH5File(filename, 'main')



def ReadGoldData(prefix):
    filename = 'gold/{0}_gold.h5'.format(prefix)

    return ReadH5File(filename, 'stack')



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