import numpy as np

from ibex.utilities.constants import *

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


class Skeleton:
    def __init__(self, prefix, label):
        self.label = label
        self.joints = []
        self.endpoints = []

        # get the skeleton filename
        filename = 'skeletons/' + prefix + '/tree_' + str(label) + '.swc'

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