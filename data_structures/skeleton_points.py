import struct
import numpy as np


from ibex.utilities.constants import *



class Skeleton:
    def __init__(self, label, joints, endpoints):
        self.label = label
        self.joints = joints
        self.endpoints = endpoints

    def NPoints(self):
        return len(self.joints) + len(self.endpoints)

    def Endpoints2Array(self):
        nendpoints = len(self.endpoints)

        array = np.zeros((nendpoints, 3), dtype=np.int64)
        for ie in range(nendpoints):
            array[ie] = self.endpoints[ie]

        return array

    def Joints2Array(self):
        njoints = len(self.endpoints) + len(self.joints)

        array = np.zeros((njoints, 3), dtype=np.int64)
        index = 0
        for endpoint in self.endpoints:
            array[index] = endpoint
            index += 1
        for joint in self.joints:
            array[index] = joint
            index += 1

        return array


    def WorldJoints2Array(self, resolution):
        njoints = len(self.endpoints) + len(self.joints)

        array = np.zeros((njoints, 3), dtype=np.int64)
        index = 0
        for endpoint in self.endpoints:
            array[index] = (endpoint[IB_Z] * resolution[IB_Z], endpoint[IB_Y] * resolution[IB_Y], endpoint[IB_X] * resolution[IB_X])
            index += 1
        for joint in self.joints:
            array[index] = (joint[IB_Z] * resolution[IB_Z], endpoint[IB_Y] * resolution[IB_Y], endpoint[IB_X] * resolution[IB_X])
            index += 1

        return array


class Skeletons:
    def __init__(self, prefix, skeleton_algorithm, downsample_resolution, benchmark, params):
        self.skeletons = []

        # read in all of the skeleton points
        if benchmark: filename = 'benchmarks/skeleton/{}-{}-{:03d}x{:03d}x{:03d}-upsample-{}-skeleton.pts'.format(prefix, skeleton_algorithm, downsample_resolution[IB_X], downsample_resolution[IB_Y], downsample_resolution[IB_Z], params)
        else: filename = 'skeletons/{}/{}-{:03d}x{:03d}x{:03d}-upsample-{}-skeleton.pts'.format(prefix, skeleton_algorithm, downsample_resolution[IB_X], downsample_resolution[IB_Y], downsample_resolution[IB_Z], params)

        with open(filename, 'rb') as fd:
            zres, yres, xres, max_label, = struct.unpack('qqqq', fd.read(32))

            for label in range(max_label):
                joints = []
                endpoints = []

                njoints, = struct.unpack('q', fd.read(8))
                for _ in range(njoints):
                    iv, = struct.unpack('q', fd.read(8))
                    
                    # endpoints are negative
                    endpoint = False
                    if (iv < 0): 
                        iv = -1 * iv 
                        endpoint = True

                    iz = iv / (yres * xres)
                    iy = (iv - iz * yres * xres) / xres
                    ix = iv % xres

                    if endpoint: endpoints.append((iz, iy, ix))
                    else: joints.append((iz, iy, ix))

                self.skeletons.append(Skeleton(label, joints, endpoints))

    def NSkeletons(self):
        return len(self.skeletons)


    def KthSkeleton(self, k):
        return self.skeletons[k]