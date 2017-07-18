from ibex.utilities.constants import *

class IBBox:
    def __init__(self, mins, maxs):
        self.mins = mins
        self.maxs = maxs

    def __str__(self):
        return '({},{},{})-({},{},{})'.format(self.mins[IB_X], self.mins[IB_Y], self.mins[IB_Z], self.maxs[IB_X], self.maxs[IB_Y], self.maxs[IB_Z])

    def Mins(self): return list(self.mins)
    def Maxs(self): return list(self.maxs)

    def Min(self, dim): return self.mins[dim]
    def Max(self, dim): return self.maxs[dim]

    def ZMin(self): return self.mins[IB_Z]
    def YMin(self): return self.mins[IB_Y]
    def XMin(self): return self.mins[IB_X]
    def ZMax(self): return self.maxs[IB_Z]
    def YMax(self): return self.maxs[IB_Y]
    def XMax(self): return self.maxs[IB_X]

    def Intersection(self, other):
        # get the new bounding box
        if (self.mins[IB_Z] < other.mins[IB_Z]): self.mins[IB_Z] = other.mins[IB_Z]
        if (self.mins[IB_Y] < other.mins[IB_Y]): self.mins[IB_Y] = other.mins[IB_Y]
        if (self.mins[IB_X] < other.mins[IB_X]): self.mins[IB_X] = other.mins[IB_X]
        if (self.maxs[IB_Z] > other.maxs[IB_Z]): self.maxs[IB_Z] = other.maxs[IB_Z]
        if (self.maxs[IB_Y] > other.maxs[IB_Y]): self.maxs[IB_Y] = other.maxs[IB_Y]
        if (self.maxs[IB_X] > other.maxs[IB_X]): self.maxs[IB_X] = other.maxs[IB_X]