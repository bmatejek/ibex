class SWCEntry:
    def __init__(self, sample_number, structure_id, x, y, z, radius, parent_sample):
        self.sample_number = sample_number
        self.structure_id = structure_id
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.parent_sample = parent_sample

    def Point(self):
        return (self.x, self.y, self.z)

class Skeleton:
    def __init__(self, prefix, label):
        self.label = label
        self.joints = []
        self.endpoints = []

        filename = 'skeletons/' + prefix + '/tree_' + str(label) + '.swc'

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

                self.joints.append(SWCEntry(sample_number, structure_id, x, y, z, radius, parent_sample))

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

        for ie in range(len(endpoints)):
            if endpoints[ie]:
                self.endpoints.append(self.joints[ie])
