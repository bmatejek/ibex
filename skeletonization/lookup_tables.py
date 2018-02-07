# import os
# import sys
# import re

# import numpy as np
# from numba import jit

# from ibex.utilities import dataIO



# def String2Mask(string):
#     # get the number of elements in the string
#     width = 3
#     nelements = width * width * width
#     assert (len(string) == nelements)
    
#     # string to number mapping
#     # '.' is anything, at least one 'x' must be 1
#     mapping = {'0': 0, '1': 1, '.': 2, 'x': 3}

#     # get the mask in a numpy array to allow for 
#     mask = np.zeros((width, width, width), dtype=np.uint8)

#     index = 0
#     for iz in range(width):
#         for iy in range(width):
#             for ix in range(width):
#                 mask[iz,iy,ix] = mapping[string[index]]
#                 index += 1

#     # return the mask as a cubic array
#     return mask



# def ReadMasks(thinning=True):
#     if thinning: prefix = 'thinning'
#     else: prefix = 'smoothing'

#     # get the directory of this ibex folder
#     directory = os.path.dirname(os.path.realpath(__file__))
#     filename = '{}/{}.txt'.format(directory, prefix)

#     base_masks = []

#     # open the file
#     with open(filename, 'r') as fd:
#         # iterate over all of the masks
#         for mask in fd.readlines():
#             # allow comments in mask files
#             if mask[0] == '#': continue
#             mask = String2Mask(mask.split(': ')[1].strip().replace(' ', ''))

#             base_masks.append(mask)

#     return base_masks



# @jit(nopython=True)
# def GetLookupValue(mask, binary):
#     width = mask.shape[0]       # get the width of this mask
#     nvariable = 0               # number of 2s or 3s seen

#     ii = 0                      # keep track of the number of elements traversed
#     index = 0                   # the index corresponding to the mask and binary value

#     # go through the entire mask and keep track 
#     for iz in range(width):
#         for iy in range(width):
#             for ix in range(width):
#                 # this bit is one
#                 if mask[iz,iy,ix] == 1: index += (1 << ii)
#                 # this bit is if the corresponding entry in binary is one
#                 elif mask[iz,iy,ix] == 2 or mask[iz,iy,ix] == 3:
#                     if (1 << nvariable) & binary: index += (1 << ii)
#                     nvariable += 1

#                 ii += 1

#     return index 




# def PopulateTable(lookup_table, mask, direction):
#     # get the number of dots and exes
#     dots = np.count_nonzero(mask == 2)
#     exes = np.count_nonzero(mask == 3)
#     assert (not dots or not exes)

#     # get the number of unique elements and the bit to flip
#     nunique = 1 << max(dots, exes)
#     bit = 1 << direction

#     # iterate through all possible binary representations for these unique values
#     min_label = (exes > 0)
#     for iv in range(min_label, nunique):
#         lookup_table[GetLookupValue(mask, iv)] += bit

#     # return the updated lookup table
#     return lookup_table



# def GenerateThinningTable(masks):
#     # get the number of directions and rotations
#     NDIRECTIONS = 6             # U, D, N, E, S, W
#     NROTATIONS = 4              # 0, 90, 180, 270

#     # generate an empty lookup table
#     lookup_table = np.zeros(2**27, dtype=np.uint8)

#     # iterate over each direction and rotation for every mask
#     for mask in masks:
#         for direction in range(NDIRECTIONS):
#             # this corresponds to the U, E, D, W planes which are rotations along yz-axis
#             if direction < 4:
#                 directed_mask = np.rot90(mask, k=direction, axes=(0,1))
#             # else rotate along the xz-axis for the N and S directions
#             elif direction == 5:
#                 directed_mask = np.rot90(mask, k=1, axes=(0,2))
#             else:
#                 directed_mask = np.rot90(mask, k=3, axes=(0,2))
            
#             # rotate the mask along the current xy-plane
#             for rotation in range(NROTATIONS):
#                 rotated_mask = np.rot90(directed_mask, k=rotation, axes=(1,2))

#                 # add all viable options to the lookup table
#                 # direction is used for which bit to flip (use one lookup table rather than 6)
#                 PopulateTable(lookup_table, rotated_mask, direction)

#     # return the full lookup table
#     return lookup_table


# def GenerateSmoothingTable(masks):
#     # get the number of directions
#     NDIRECTIONS = 2             # R1, R2
#     width = masks[0].shape[0]

#     # generate an empty lookup table
#     lookup_table = np.zeros(2**27, dtype=np.uint8)

#     # iterate over each direction 
#     for mask in masks:
#         for direction in range(NDIRECTIONS):
#             if direction == 0:
#                 directed_mask = np.copy(mask)
#             else:
#                 # reverse direction
#                 flattened_mask = mask.flatten()
#                 reversed_mask = np.flipud(flattened_mask)
#                 directed_mask = np.reshape(reversed_mask, (width, width, width))

#             # add all viable options to the lookup table
#             # direction is used for which bit to flip (use one lookup)
#             PopulateTable(lookup_table, directed_mask, direction)

#     # return the full lookup table
#     return lookup_table



# def TryCache(prefix):
#     directory = os.path.dirname(os.path.realpath(__file__))
#     cache_filename = '{}/{}-cache.npy'.format(directory, prefix)
    
#     return None

#     if os.path.exists(cache_filename):
#         return np.load(cache_filename)
#     else:
#         return None


# def GetRegularExpresions(prefix):
#     # get the directory of this ibex folder
#     directory = os.path.dirname(os.path.realpath(__file__))
#     filename = '{}/{}.txt'.format(directory, prefix)   
   
#     # get all possible masks
#     simple_masks = []
#     complex_masks = []

#     # open the file
#     with open(filename, 'r') as fd:
#         # iterate over all masks
#         for mask in fd.readlines():
#             # allow comments in mask files
#             if mask[0] == '#': continue
#             mask = mask.split(': ')[1].strip().replace(' ', '')

#             if 'x' in mask:
#                 regex = re.compile(mask.replace('x', '.'))
#                 complex_masks.append((regex, mask))
#             else:
#                 simple_masks.append(re.compile(mask))

#     return simple_masks, complex_masks



# jit(nopython=True)
# def FindMatch(deletable_point, simple_masks, complex_masks):    
#     # go through the simple masks
#     for regex in simple_masks:
#         if regex.search(deletable_point) is not None: return True


#     # go through complex masks where one 'x' must be 1
#     for (regex, mask) in complex_masks:
#         if regex.search(deletable_point) is not None:
#             # make sure at least one 'x' is 1
#             found_one_x = False
#             for iv in range(27):
#                 if mask[iv] == 'x' and deletable_point[iv] == '1': return True

#     # no match found
#     return False



# def ValidateSmoothingLookupTable(lookup_table):
#     simple_masks, complex_masks = GetRegularExpresions('smoothing')
#     NDIRECTIONS = 2             # R1, R2

#     # go through all values in the lookup table
#     for iv in range(lookup_table.size):
#         if (iv % 100000 == 0): print iv
#         # write this number as a binary string 
#         deletable_point = '{:027b}'.format(iv)

#         # make sure there is a match to at least one lookup table
#         for direction in range(NDIRECTIONS):
#             # (need to reverse for R1 since we want low digits first)
#             if direction == 0: deletable_point = deletable_point[::-1]

#             # do we include this as a match
#             if lookup_table[iv] & (1 << direction):
#                 assert (FindMatch(deletable_point, simple_masks, complex_masks))
#             else:
#                 assert (not FindMatch(deletable_point, simple_masks, complex_masks))



# def ValidateThinningLookupTable(lookup_table):
#     simple_masks, complex_masks = GetRegularExpresions('thinning')
#     NDIRECTIONS = 6             # U, D, N, S, E, W

#     # go through all values in the lookup table
#     for iv in range(lookup_table.size):
#         if (iv % 1000000 == 0): print iv
#         # write this number as a binary string
#         deletable_point = '{:027b}'.format(iv)

#         # make sure there is a match in at least one lookup table
#         for direction in range(NDIRECTIONS):
#             # write this number as 



# def GenerateLookupTables():
#     directory = os.path.dirname(os.path.realpath(__file__))

#     # try the cache, otherwise create the table
#     # thinning_lookup_table = TryCache('thinning')
#     # if thinning_lookup_table is None:
#     #     thinning_masks = ReadMasks(thinning=True)
#     #     thinning_lookup_table = GenerateThinningTable(thinning_masks)
        
#     #     ValidateThinningLookupTable('thinning', thinning_lookup_table)

#         # cache_filename = '{}/thinning-cache.npy'.format(directory)
#         # np.save(cache_filename, thinning_lookup_table)
 
#     smoothing_lookup_table = TryCache('smoothing')
#     if smoothing_lookup_table is None:
#         smoothing_masks = ReadMasks(thinning=False)
#         smoothing_lookup_table = GenerateSmoothingTable(smoothing_masks)

#         ValidateSmoothingLookupTable(smoothing_lookup_table)

#         # cache_filename = '{}/smoothing-cache.npy'.format(directory)
#         # np.save(cache_filename, smoothing_lookup_table)

#     return thinning_lookup_table, smoothing_lookup_table