import struct
import numpy as np


from ibex.utilities.constants import *



class Skeleton:
    def __init__(self, label, joints, endpoints):
        self.label = label
        self.joints = joints
        self.endpoints = endpoints

    def Endpoints2Array(self):
        nendpoints = len(self.endpoints)

        array = np.zeros((nendpoints, 3), dtype=np.int64)
        for ie in range(nendpoints):
            array[ie] = self.endpoints[ie]

        return array


class Skeletons:
    def __init__(self, prefix, skeleton_algorithm='thinning', downsample_resolution=(100, 100, 100), benchmark=False):
        self.skeletons = []

        # read in all of the skeleton points
        if benchmark: filename = 'benchmarks/skeleton/{}-topological-{}x{}x{}-{}-skeleton.pts'.format(prefix, downsample_resolution[IB_X], downsample_resolution[IB_Y], downsample_resolution[IB_Z], skeleton_algorithm)
        else: filename = 'skeletons/{}/topological-{}x{}x{}-{}-skeleton.pts'.format(prefix, downsample_resolution[IB_X], downsample_resolution[IB_Y], downsample_resolution[IB_Z], skeleton_algorithm)

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