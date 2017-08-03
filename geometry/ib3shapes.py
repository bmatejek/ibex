from copy import deepcopy
from ibex.utilities.constants import *

class IBBox:
    def __init__(self, mins, maxs):
        self.mins = list(mins)
        self.maxs = list(maxs)

    def __str__(self):
        return '({},{},{})-({},{},{})'.format(self.mins[IB_X], self.mins[IB_Y], self.mins[IB_Z], self.maxs[IB_X], self.maxs[IB_Y], self.maxs[IB_Z])

    def Intersection(self, other):
        # get the new bounding box
        if (self.mins[IB_Z] < other.mins[IB_Z]): self.mins[IB_Z] = other.mins[IB_Z]
        if (self.mins[IB_Y] < other.mins[IB_Y]): self.mins[IB_Y] = other.mins[IB_Y]
        if (self.mins[IB_X] < other.mins[IB_X]): self.mins[IB_X] = other.mins[IB_X]
        if (self.maxs[IB_Z] > other.maxs[IB_Z]): self.maxs[IB_Z] = other.maxs[IB_Z]
        if (self.maxs[IB_Y] > other.maxs[IB_Y]): self.maxs[IB_Y] = other.maxs[IB_Y]
        if (self.maxs[IB_X] > other.maxs[IB_X]): self.maxs[IB_X] = other.maxs[IB_X]
