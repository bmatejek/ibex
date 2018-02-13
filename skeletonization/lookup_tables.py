import os
import sys
import re

import numpy as np
from numba import jit

from ibex.utilities import dataIO



def TryCache():
    directory = os.path.dirname(os.path.realpath(__file__))
    r1_filename = '{}/lut_smoothing_r1.dat'
    r2_filename = '{}/lut_smoothing_r2.dat'

    # if these files exist, don't need to generate them
    if os.path.exists(r1_filename) and os.path.exist(r2_filename): return True
    else: return False



def String2Mask(string):
    # get the number of elements in the string
    nelements = len(string)

    # string to number mapping
    # '.' is anything, at least one 'x' must be 1
    mapping = {'0': 0, '1': 1, '.': 2, 'x': 3}

    # get the mask in a numpy array
    mask = np.zeros(nelements, dtype=np.uint8)

    for index in range(nelements):
        mask[index] = mapping[string[index]]

    # return the mask as a cubic array
    return mask



def ReadMasks():
    directory = os.path.dirname(os.path.realpath(__file__))
    filename = '{}/smoothing.txt'.format(directory)

    base_masks = []

    # open the file
    with open(filename, 'r') as fd:
        # iterate over all masks
        for mask in fd.readlines():
            # allow comments in mask files
            if mask[0] == '#': continue
            mask = String2Mask(mask.split(': ')[1].strip().replace(' ', ''))

            base_masks.append(mask)

    return base_masks



@jit(nopython=True)
def GetLookupValue(mask, binary):
    width = mask.shape[0]       # get the width of this mask
    nvariable = 0               # number of 2s or 3s seen

    ii = 0                      # keep track of the number of elements traversed
    value = 0                   # binary value from this mask

    # go through mask
    for iz in range(width):
        for iy in range(width):
            for ix in range(width):
                # skip center value 
                if (iz == width / 2 and iy == width / 2 and ix == width / 2): continue
                # this bit is always 1
                if mask[iz,iy,ix] == 1: value += (1 << ii)
                # this bit is 1 if corresponding entry in binary is 1
                elif mask[iz,iy,ix] == 2 or mask[iz,iy,ix] == 3:
                    if (1 << nvariable) & binary: value += (1 << ii)
                    nvariable += 1

                ii += 1

    return value



def PopulateTable(lookup_table, mask):
    # get the number of dots and exes
    dots = np.count_nonzero(mask == 2)
    exes = np.count_nonzero(mask == 3)
    assert (not dots or not exes)

    # get the number of unique elements and the bit to flip
    nunique = 1 << max(dots, exes)

    # iterate through all binary representations for unique values
    min_label = (exes > 0)
    for iv in range(min_label, nunique):
        value = GetLookupValue(mask, iv)
        assert ((0 <= value) and (value < 2**26))
        offset = value % 8
        index = value / 8
        assert ((0 <= index) and (index < 2**23))
        lookup_table[index] = lookup_table[index] + 2**offset


def GenerateSmoothingLookupTable():
    # see if the files exist
    if TryCache(): return True

    # create empty unsigned char arrays with 2 ** 23 elements
    # this corresponds to 2 ** 26 bits
    lut_smoothing_r1 = np.zeros(2**23, dtype=np.uint8)
    lut_smoothing_r2 = np.zeros(2**23, dtype=np.uint8)

    smoothing_masks = ReadMasks()

    # iterate over every mask
    for mask in smoothing_masks:
        # consider the r0 and r1 direction
        for direction in ['r1', 'r2']:
            if direction == 'r1':
                directed_mask = np.reshape(mask, (3, 3, 3))
                # add all viable options to the lookup table
                PopulateTable(lut_smoothing_r1, directed_mask)
            else:
                reversed_mask = np.flipud(mask)
                directed_mask = np.reshape(reversed_mask, (3, 3, 3))
                # add all viable options to the lookup table
                PopulateTable(lut_smoothing_r2, directed_mask)
    
    directory = os.path.dirname(os.path.realpath(__file__))
    r1_filename = '{}/lut_smoothing_r1.dat'.format(directory)
    r2_filename = '{}/lut_smoothing_r2.dat'.format(directory)

    with open(r1_filename, 'wb') as fd:
        fd.write(lut_smoothing_r1)
    with open(r2_filename, 'wb') as fd:
        fd.write(lut_smoothing_r2)