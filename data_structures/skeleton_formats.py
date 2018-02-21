import struct
import numpy as np


from ibex.utilities.constants import *

class TopologyEntry:
    def __init__(self, skeleton, x, y, z, label):
        self.skeleton = skeleton
        self.x = x
        self.y = y
        self.z = z
        self.label = label
    
    def GridPoint(self):
        grid_location = (self.z, self.y, self.x)
        return np.array(grid_location).astype(dtype=np.int32)

    def WorldPoint(self, world_res):
        world_location = (self.z * world_res[IB_Z], self.y * world_res[IB_Y], self.x * world_res[IB_X])
        return np.array(world_location).astype(dtype=np.int32)
    
    def X(self):
        return self.x

    def Y(self):
        return self.y

    def Z(self):
        return self.z

    def Label(self):
        return self.label



class TopologySkeleton:
    def __init__(self, prefix, label, resolution):
        self.label = label
        self.joints = []
        self.endpoints = []

        # skip extracellular predictions
        if label == 0: return

        # get the resolution in each dimension
        zres = resolution[IB_Z]
        yres = resolution[IB_Y]
        xres = resolution[IB_X]

        # get the skeleton filename
        filename = 'skeletons/topological-thinning/{}/skeleton-{}.pts'.format(prefix, label)

        # open the file and read all of the points
        with open(filename, 'rb') as fd:
            npts, = struct.unpack('l', fd.read(8))

            for _ in range(npts):
                index, = struct.unpack('l', fd.read(8))
                    
                # is this index an endpoint
                endpoint = False
                if (index < 0):
                    endpoint = True
                    index *= -1

                iz = index / (xres * yres)
                iy = (index - iz * xres * yres) / xres
                ix = index % xres

                skeleton_entry = TopologyEntry(self, ix, iy, iz, label)
                self.joints.append(skeleton_entry)
                if endpoint: self.endpoints.append(skeleton_entry)         

    # return the kth endpoint
    def EndPoint(self, k):
        assert (0 <= k and k < self.NEndPoints())
        return self.endpoints[k]

    # return all endpoints
    def EndPoints(self):
        return self.endpoints

    # return the number of endpoints
    def NEndPoints(self):
        return len(self.endpoints)

    # return the kth joint
    def Joint(self, k):
        assert (0 <= k and k < self.NJoints())
        return self.joints[k]

    # return all joints
    def Joints(self):
        return self.joints

    # return the number of joints
    def NJoints(self):
        return len(self.joints)

    # return the segment label
    def Label(self):
        return self.label
        


class SWCEntry:
    def __init__(self, skeleton, sample_number, structure_id, x, y, z, radius, parent_sample, label):
        self.skeleton = skeleton
        self.sample_number = sample_number
        self.structure_id = structure_id
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.parent_sample = parent_sample
        self.label = label

    def GridPoint(self):
        grid_location = (self.z, self.y, self.x)
        return np.array(grid_location).astype(dtype=np.int32)

    def SampleNumber(self):
        return self.sample_number

    def StructureID(self):
        return self.structure_id

    def X(self):
        return self.x

    def Y(self):
        return self.y

    def Z(self):
        return self.z

    def Radius(self):
        return self.radius

    def ParentSample(self):
        return self.parent_sample

    def Label(self):
        return self.label

    def WorldPoint(self, world_res):
        world_location = (self.z * world_res[IB_Z], self.y * world_res[IB_Y], self.x * world_res[IB_X])
        return np.array(world_location).astype(dtype=np.int32)

    def Parent(self):
        # return parent sample if it exists, otherwise return child
        if not self.parent_sample == -1: return self.skeleton.joints[self.parent_sample - 1]
        # othersiw find a child whose parent is this node
        for joint in self.skeleton.joints:
            if joint.parent_sample == self.sample_number: return joint

    def Neighbors(self):
        neighbors = []

        # return all of the neighbor coordinates next to this entry
        for joint in self.skeleton.joints:
            if self.parent_sample == joint.sample_number:
                neighbors.append(joint)
            elif joint.parent_sample == self.sample_number:
                neighbors.append(joint)

        return neighbors


class SWCSkeleton:
    def __init__(self, prefix, minObjSize, minLength, label):
        self.label = label
        self.joints = []
        self.endpoints = []

        # get the skeleton filename
        filename = 'skeletons/NeuTu/{}-{}-{}/tree_{}.swc'.format(prefix, minObjSize, minLength, label)

        # open the file and read all of the swc entries
        with open(filename, 'r') as fd:
            for joint in fd.readlines():
                # ignore comments
                if (joint[0] == '#'): continue

                # split the attributes
                attributes = joint.split(' ')

                # generate an SWC Entry
                sample_number = int(attributes[0])
                structure_id = int(attributes[1])
                x = float(attributes[2])
                y = float(attributes[3])
                z = float(attributes[4])
                radius = float(attributes[5])
                parent_sample = int(attributes[6])

                self.joints.append(SWCEntry(self, sample_number, structure_id, x, y, z, radius, parent_sample, label))

        # go through every joint to determine endpoints
        endpoints = [True] * len(self.joints)

        # if the joint has a parent that parent is not an endpoint
        for joint in self.joints:
            if not joint.parent_sample == -1:
                endpoints[joint.parent_sample - 1] = False

        # if a joint has no parent it is an endpoint
        for ie, joint in enumerate(self.joints):
            if joint.parent_sample == -1:
                endpoints[ie] = True

        # create a list of endpoints 
        for ie in range(len(endpoints)):
            if endpoints[ie]:
                self.endpoints.append(self.joints[ie])

    # return the kth endpoint
    def EndPoint(self, k):
        assert (0 <= k and k < self.NEndPoints())
        return self.endpoints[k]

    # return all endpoints
    def EndPoints(self):
        return self.endpoints

    # return the number of endpoints
    def NEndPoints(self):
        return len(self.endpoints)

    # return the kth joint
    def Joint(self, k):
        assert (0 <= k and k < self.NJoints())
        return self.joints[k]

    # return all joints
    def Joints(self):
        return self.joints

    # return the number of joints
    def NJoints(self):
        return len(self.joints)

    # return the segment label
    def Label(self):
        return self.label