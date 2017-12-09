import math
import struct
import numpy as np
import time
import scipy.ndimage
import skimage.transform
import os.path

from scipy.spatial import KDTree
from numba import jit

from ibex.utilities import dataIO
from ibex.super_resolution.util import *
from ibex.super_resolution import visualization

# for KDTree to work
import sys
sys.setrecursionlimit(100000)


# convert the RGB image to YIQ
def RGB2YIQ(image):
    Y = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    I = 0.596 * image[:,:,0] - 0.274 * image[:,:,1] - 0.322 * image[:,:,2]
    Q = 0.211 * image[:,:,0] - 0.523 * image[:,:,1] + 0.312 * image[:,:,2]
    return Y, I, Q

# convert the YIQ images to RGB
def YIQ2RGB(Y, I, Q):
    yres, xres = Y.shape
    RGB = np.zeros((yres, xres, 3), dtype=np.float32)
    RGB[:,:,0] = Y + 0.956 * I + 0.621 * Q
    RGB[:,:,1] = Y - 0.272 * I - 0.647 * Q
    RGB[:,:,2] = Y - 1.106 * I + 1.703 * Q
    return RGB

# remove average intensity from image
def RemoveDC(intensity):
    return intensity - np.mean(intensity)


# create the kdtree of features for this image
def CreateKDTree(intensities, diameter):
    # get useful parameters
    yres, xres = intensities.shape
    nfeatures = yres * xres
    feature_size = diameter * diameter
    features = np.zeros((nfeatures, feature_size), dtype=np.float32)
    for iy in range(yres):
        for ix in range(xres):
            features[IndicesToIndex(iy, ix, xres),:] = ExtractFeature(intensities, iy, ix, diameter)

    return KDTree(features)

# get the patch-specific good distances
def GoodDistance(intensities, diameter):
    # get useful parameters
    yres, xres = intensities.shape
    # shift the image by half a pixel in a diagonal direction
    pixel_shift = 1.0 / (2 * math.sqrt(2))
    shifts = scipy.ndimage.interpolation.shift(intensities, (pixel_shift, pixel_shift), order=3)
    distances = np.zeros((yres, xres), dtype=np.float32)

    for iy in range(yres):
        for ix in range(xres):
            intensity_feature = ExtractFeature(intensities, iy, ix, diameter)
            shift_feature = ExtractFeature(shifts, iy, ix, diameter)

            distances[iy,ix] = math.sqrt(np.sum(np.multiply(intensity_feature - shift_feature, intensity_feature - shift_feature)))

    return distances, shifts

# create a hierarchy of RGBs, intensities, and kdtrees
def CreateHierarchy(parameters, hierarchies):
    # get useful parameters
    root_filename = parameters['root_filename']
    m = parameters['m']                             # number of down level supports
    n = parameters['n']                             # number of up scalings
    scale = parameters['scale']                     # scale to increase each layer
    truncate = parameters['truncate']               # number of sigma in gaussian blur
    diameter = parameters['diameter']               # diameter of features

    RGBs = hierarchies['RGBs']
    Ys = hierarchies['Ys']
    Is = hierarchies['Is']
    Qs = hierarchies['Qs']
    features = hierarchies['features']
    kdtrees = hierarchies['kdtrees']
    distances = hierarchies['distances']
    shifts = hierarchies['shifts']

    # read in the first image and convert to YIQ
    filename = 'pictures/{}.png'.format(root_filename)
    RGBs[0] = dataIO.ReadImage(filename)
    Ys[0], Is[0], Qs[0] = RGB2YIQ(RGBs[0])
    
    yres, xres = Ys[0].shape

    # output the input image into the hierarchy subfolder
    intensity_filename = 'hierarchy/{}-intensity-0.png'.format(root_filename)
    rgb_filename = 'hierarchy/{}-RGB-0.png'.format(root_filename)
    dataIO.WriteImage(intensity_filename, Ys[0])
    dataIO.WriteImage(rgb_filename, RGBs[0])

    # go through all levels of the hierarchy
    for iv in range(-1, -(m + 1), -1):
        # get the standard deviation of the blur filter
        sigma = math.sqrt(pow(scale, -iv))

        # filter the input image 
        blurred_intensity = scipy.ndimage.filters.gaussian_filter(Ys[0], sigma, truncate=truncate)
        blurred_I = scipy.ndimage.filters.gaussian_filter(Is[0], sigma, truncate=truncate)
        blurred_Q = scipy.ndimage.filters.gaussian_filter(Qs[0], sigma, truncate=truncate)
        blurred_red = scipy.ndimage.filters.gaussian_filter(RGBs[0][:,:,0], sigma, truncate=truncate)
        blurred_green = scipy.ndimage.filters.gaussian_filter(RGBs[0][:,:,1], sigma, truncate=truncate)
        blurred_blue = scipy.ndimage.filters.gaussian_filter(RGBs[0][:,:,2], sigma, truncate=truncate)

        # get the scale difference between this level and the first level
        magnification = pow(scale, iv)
        low_yres, low_xres = (ceil(yres * magnification), ceil(xres * magnification))
        
        # perform bilinear interpolation
        Ys[iv] = skimage.transform.resize(blurred_intensity, (low_yres, low_xres), order=1, mode='reflect')
        Is[iv] = skimage.transform.resize(blurred_I, (low_yres, low_xres), order=1, mode='reflect')
        Qs[iv] = skimage.transform.resize(blurred_Q, (low_yres, low_xres), order=1, mode='reflect')
        RGBs[iv] = np.zeros((low_yres, low_xres, 3), dtype=np.float32)
        RGBs[iv][:,:,0] = skimage.transform.resize(blurred_red, (low_yres, low_xres), order=1, mode='reflect')
        RGBs[iv][:,:,1] = skimage.transform.resize(blurred_green, (low_yres, low_xres), order=1, mode='reflect')
        RGBs[iv][:,:,2] = skimage.transform.resize(blurred_blue, (low_yres, low_xres), order=1, mode='reflect')

        # output the images
        intensity_filename = 'hierarchy/{}-intensity-{}.png'.format(root_filename, -iv)
        rgb_filename = 'hierarchy/{}-RGB-{}.png'.format(root_filename, -iv)
        dataIO.WriteImage(intensity_filename, Ys[iv])
        dataIO.WriteImage(rgb_filename, RGBs[iv])

    # the upsampling of I and Q is independent of super resolution
    for iv in range(1, n + 1):
        magnification = pow(scale, iv)
        high_yres, high_xres = (ceil(yres * magnification), ceil(xres * magnification))
        Is[iv] = skimage.transform.resize(Is[0], (high_yres, high_xres), order=3, mode='reflect')
        Qs[iv] = skimage.transform.resize(Qs[0], (high_yres, high_xres), order=3, mode='reflect')

    # remove all DC components from intesity images for feature extraction
    for iv in range(0, -(m + 1), -1):
        features[iv] = RemoveDC(Ys[iv])
        kdtrees[iv] = CreateKDTree(features[iv], diameter)
        distances[iv], shifts[iv] = GoodDistance(features[iv], diameter)

# create a gaussian kernel
def Gaussian(variance, truncate):
    # no variance equals the delta function
    if variance == 0: return np.ones((1, 1), dtype=np.float32)

    sigma = math.sqrt(variance)
    
    # get the radius that ends after truncate deviations
    radius = int(truncate * sigma + 0.5)

    # create a mesh grid of size (2 * radius + 1)^2
    span = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(span, span)

    # create the kernel
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)) 
    kernel = kernel / np.sum(kernel)

    return kernel

# class Constraint:
#     def __init__(self, iy, ix, yoffset, xoffset, level, value):
#         self.iy = iy
#         self.ix = ix
#         self.yoffset = yoffset
#         self.xoffset = xoffset
#         self.level= level
#         self.value = value

# # save the constraints file
# def SaveConstraints(constraints_filename, constraints):
#     # save the constraints to the cache
#     with open(constraints_filename, 'wb') as fd:
#         fd.write(struct.pack('i', len(constraints)))
#         for constraint in constraints:
#             fd.write(struct.pack('iiiiif', constraint.iy, constraint.ix, constraint.yoffset, constraint.xoffset, constraint.level, constraint.value))

# # read the constraints file
# def ReadConstraints(constraints_filename):
#     constraints = []
#     with open(constraints_filename, 'rb') as fd:
#         nconstraints, = struct.unpack('i', fd.read(4))
#         for iv in range(nconstraints):
#             iy, ix, yoffset, xoffset, level, value, = struct.unpack('iiiiif', fd.read(24))
#             constraints.append(Constraint(iy, ix, yoffset, xoffset, level, value))

#     return constraints

# # increase the resolution to level in one pass
# def SinglePassSuperResolution(parameters, hierarchies, n):

    
#     L = Ys[0]
#     Lfeatures = features[0]
#     Ldist = distances[0]
#     Lyres, Lxres = L.shape
#     Lradius = (diameter[0] / 2, diameter[1] / 2)
#     Lkdtree = kdtrees[0]

#     magnification = pow(scale, n)

#     high_yres, high_xres = (ceil(Lyres * magnification), ceil(Lxres * magnification))
#     highres_Y = np.zeros((high_yres, high_xres), dtype=np.float32)

#     # create a hierarchy of gaussian blurs
#     blurs = [Gaussian(pow(scale, iv), truncate) for iv in range(n + 1)]

#     # read this file    
#     cache_filename = 'cache/{}-classical-super-resolution-constraints.cache'.format(root_filename)

#     nnonzero = 0
#     if os.path.exists(cache_filename):
#         constraints = ReadConstraints(cache_filename)
#     else:
#         # create a set of constraints
#         constraints = []

#         level = 0
#         # get the single image SR components
#         for iy in range(Lradius[0], Lyres - Lradius[0]):
#             start_time = time.time()
#             for ix in range(Lradius[1], Lxres - Lradius[1]):
#                 # get the feature at this location
#                 feature = ExtractFeature(Lfeatures, iy, ix, diameter)

#                 values, locations = Lkdtree.query(feature, k=8, distance_upper_bound=Ldist[iy,ix])

#                 for value, location in zip(values, locations):
#                     Ly, Lx = IndexToIndices(location, Lxres)
#                     if Ly == iy and Lx == ix: 
#                         # add the constraint
#                         constraints.append(Constraint(iy, ix, 0, 0, 0, L[Ly,Lx]))
#                         continue
#                     if value == float('inf'): continue

#                     # check the four possible shifts at this scale level
#                     best_match = Ldist[iy,ix]
#                     yoffset = 0
#                     xoffset = 0
#                     subpixel = 1.0 / magnification
#                     for xshift in [-subpixel, 0, subpixel]:
#                         for yshift in [-subpixel, 0, subpixel]:
#                             # get the shifted location
#                             shift = scipy.ndimage.interpolation.shift(Lfeatures[Ly-Lradius[0]:Ly+Lradius[0]+1,Lx-Lradius[1]:Lx+Lradius[1]+1], (yshift, xshift))
#                             shift_feature = np.multiply(ssd_kernel, shift.flatten())
#                             # get the difference in the alignments
#                             difference = math.sqrt(np.sum(np.multiply(feature - shift_feature, feature - shift_feature)))

#                             # update the best offset
#                             if (difference < best_match):
#                                 yoffset = int(round(yshift * magnification))
#                                 xoffset = int(round(xshift * magnification))
#                                 best_match = difference
#                     if yoffset == 0 and xoffset == 0: continue
#                     # add the constraint
#                     constraints.append(Constraint(iy, ix, yoffset, xoffset, 0, L[Ly,Lx]))

#             print '{} completed in {} seconds'.format(iy, time.time() - start_time)

#         SaveConstraints(cache_filename, constraints)

#     # create a system of sparse equations
#     nconstraints = len(constraints)
#     nnonzero = nconstraints * blurs[n].size

#     data = np.zeros(nnonzero, dtype=np.float32)
#     i = np.zeros(nnonzero, dtype=np.float32)
#     j = np.zeros(nnonzero, dtype=np.float32)

#     b = np.zeros(nconstraints, dtype=np.float32)

#     index = 0
    
#     # go through every constraint
#     for ie, constraint in enumerate(constraints):    
#         iy = constraint.iy
#         ix = constraint.ix
#         yoffset = constraint.iy
#         xoffset = constraint.ix
#         level = constraint.level

#         # get the high resolution location
#         magnification = pow(scale, n)
#         highy, highx = (int(round(iy * magnification)) + yoffset, int(round(ix * magnification)) + xoffset)

#         # take a gaussian blur around the location in the low resolution image
#         gaussian_blur = blurs[n - level]
#         gaussian_radius = (gaussian_blur.shape[0] / 2, gaussian_blur.shape[1] / 2)

#         for ij, iv in enumerate(range(highy - gaussian_radius[0], highy + gaussian_radius[0] + 1)):
#             for ii, iu in enumerate(range(highx - gaussian_radius[1], highx + gaussian_radius[1] + 1)):
#                 if iv < 0 or iv > high_yres - 1: continue
#                 if iu < 0 or iu > high_xres - 1: continue
#                 # get this index
#                 high_index = IndicesToIndex(iv, iu, high_xres)

#                 # set the sparse parameters
#                 data[index] = gaussian_blur[ij,ii]
#                 i[index] = ie
#                 j[index] = high_index
#                 index += 1

#         # set the b value
#         b[ie] = constraint.value

#     sparse_matrix = scipy.sparse.coo_matrix((data, (i, j)), shape=(nconstraints, high_yres * high_xres))
#     H, _, _, _,_, _, _, _, _, _  = scipy.sparse.linalg.lsqr(sparse_matrix, b, show=True)

#     smally, smallx = IndexToIndices(int(round(np.amin(j))), high_xres)
#     largey, largex = IndexToIndices(int(round(np.amax(j))), high_xres)

#     maximum = np.amax(H)
#     minimum = np.amin(H)
#     print 'Minimum Index: ({}, {})'.format(smally, smallx)
#     print 'Maximum Index: ({}, {})'.format(largey, largex)
#     print 'Minimum Data: {}'.format(np.amin(data))
#     print 'Maximum Data: {}'.format(np.amax(data))
#     print 'Minumum B: {}'.format(np.amin(b))
#     print 'Maximum B: {}'.format(np.amax(b))
#     print 'Minimum: {}'.format(np.amin(H))
#     print 'Maximum: {}'.format(np.amax(H))    

#     for iy in range(high_yres):
#         for ix in range(high_xres):
#             index = IndicesToIndex(iy, ix, high_xres)
#             highres_Y[iy,ix] = (H[index] - minimum) / (maximum - minimum)

#     Ys[n] = highres_Y
#     dataIO.WriteImage('output-intensity.png', Ys[n])
#     RGBs[n] = YIQ2RGB(Ys[n], Is[n], Qs[n])
#     dataIO.WriteImage('output-classical-super-resolution.png', RGBs[n])

# example based constraints
class ExemplarBasedConstraints:
    def __init__(self, iy, ix, value, weight):
        self.iy = iy
        self.ix = ix
        self.value = value
        self.weight = weight

# apply the super resolution algorithm
def SuperResolution(parameters, hierarchies, n):
    # get useful parameters
    root_filename = parameters['root_filename']
    m = parameters['m']
    k = parameters['k']
    scale = parameters['scale']
    diameter = parameters['diameter']
    truncate = parameters['truncate']

    RGBs = hierarchies['RGBs']
    Ys = hierarchies['Ys']
    Is = hierarchies['Is']
    Qs = hierarchies['Qs']
    kdtrees = hierarchies['kdtrees']
    distances = hierarchies['distances']
    features = hierarchies['features']
    shifts = hierarchies['shifts']

    # get parameters of the high resolution image
    yres, xres = Ys[0].shape
    magnification = pow(scale, n)
    high_yres, high_xres = (ceil(magnification * yres), ceil(magnification * xres))
    highres_Y = np.zeros((high_yres, high_xres), dtype=np.float32)
    
    classical_constraints = []


    # example_constraints = []

    # # go through all levels from 0 to (n - 1)
    # for level in range(1):
    #     mid_image = Ys[level]
    #     mid_yres, mid_xres = mid_image.shape

    #     # get the magnification from mid level to high level
    #     magnification = pow(scale, n - level)

    #     # iterate through every pixel in this midres image
    #     for iy in range(mid_yres):
    #         for ix in range(mid_xres):

    #             # extract the feature at this location
    #             feature = ExtractFeature(features[level], iy, ix, diameter)
    #             max_distance = distances[level][iy,ix]

    #             # look at the kdtree in the low resolution image
    #             value, location = kdtrees[level - n].query(feature, k=1, distance_upper_bound=max_distance)
    #             if value == float('inf'): continue
    #             lowy, lowx = IndexToIndices(location, features[level - n].shape[1])

    #             # where does this patch occur in the middle resolution image
    #             midy, midx = (int(round(magnification * lowy)), int(round(magnification * lowx)))

    #             # where does (iy, ix) occur in the high resolution image
    #             highy, highx = (int(round(magnification * iy)), int(round(magnification * ix)))

    #             # the high resolution diameter
    #             high_diameter = floor(magnification * diameter)
    #             high_radius = high_diameter / 2

    #             # get the weight of this patch depending on how good it is
    #             weight = (max_distance - value) / max_distance

    #             # create constraints for the variables
    #             for iv in range(-high_radius, high_radius + 1):
    #                 for iu in range(-high_radius, high_radius + 1):
    #                     if midy + iv < 0 or midx + iu < 0: continue
    #                     if midy + iv > mid_yres - 1 or midx + iu > mid_xres - 1: continue
    #                     if highy + iv < 0 or highx + iu < 0: continue
    #                     if highy + iv > high_yres - 1 or highx + iu > high_xres - 1: continue

    #                     mid_value = mid_image[midy+iv,midx+iu]
                        
    #                     example_constraints.append(ExemplarBasedConstraints(highy+iv, highx+iu, mid_value, weight))

    # # create the system of linear equations
    # nconstraints = len(example_constraints)
    # data = np.zeros(nconstraints, dtype=np.float32)
    # i = np.zeros(nconstraints, dtype=np.float32)
    # j = np.zeros(nconstraints, dtype=np.float32)
    # b = np.zeros(nconstraints, dtype=np.float32)

    # # populate the series of equations
    # for ie, constraint in enumerate(example_constraints):
    #     iy, ix = (constraint.iy, constraint.ix)
    #     index = IndicesToIndex(iy, ix, high_xres)

    #     i[ie] = ie
    #     j[ie] = index
    #     data[ie] = constraint.weight

    #     # set the value of the system
    #     b[ie] = constraint.weight * constraint.value

    # print '{} {}'.format(np.amin(i), np.amax(i))
    # print '{} {}'.format(np.amin(j), np.amax(j))

    # # create the sparse matrix
    # sparse_matrix = scipy.sparse.coo_matrix((data, (i, j)), shape=(nconstraints, high_yres * high_xres))
    # H, _, _, _,_, _, _, _, _, _  = scipy.sparse.linalg.lsqr(sparse_matrix, b, show=True)

    # smally, smallx = IndexToIndices(int(round(np.amin(j))), high_xres)
    # largey, largex = IndexToIndices(int(round(np.amax(j))), high_xres)

    # maximum = np.amax(H)
    # minimum = np.amin(H)

    # print 'Minimum Index: ({}, {})'.format(smally, smallx)
    # print 'Maximum Index: ({}, {})'.format(largey, largex)
    # print 'Minimum Data: {}'.format(np.amin(data))
    # print 'Maximum Data: {}'.format(np.amax(data))
    # print 'Minumum B: {}'.format(np.amin(b))
    # print 'Maximum B: {}'.format(np.amax(b))
    # print 'Minimum H: {}'.format(np.amin(H))
    # print 'Maximum H: {}'.format(np.amax(H))    

    # for iy in range(high_yres):
    #     for ix in range(high_xres):
    #         index = IndicesToIndex(iy, ix, high_xres)
    #         highres_Y[iy,ix] = (H[index] - minimum) / (maximum - minimum)

    # Ys[n] = highres_Y
    # dataIO.WriteImage('output-example-super-resolution-intensity-{}.png'.format(n), Ys[n])
    # RGBs[n] = YIQ2RGB(Ys[n], Is[n], Qs[n])
    # dataIO.WriteImage('output-example-super-resolution-{}.png'.format(n), RGBs[n])  

    # features[n] = RemoveDC(Ys[n])
    # kdtrees[n] = CreateKDTree(features[n], diameter)
    # distances[n], shifts[n] = GoodDistance(features[n], diameter)

# increase the resolution of the image
def SingleImageSR(parameters):
    # get useful parameters
    root_filename = parameters['root_filename']
    n = parameters['n']                             # number of levels to increase
    m = parameters['m']                             # number of down level supports
    scale = parameters['scale']                     # scale to increase each layer

    # useful variables
    nlayers = n + m + 1
    
    # hierarchy variables
    RGBs = [ _ for _ in range(nlayers)]
    Ys = [ _ for _ in range(nlayers)]
    Is = [ _ for _ in range(nlayers)]
    Qs = [ _ for _ in range(nlayers)]

    features = [ _ for _ in range(nlayers)]
    kdtrees = [ _ for _ in range(nlayers)]
    distances = [ _ for _ in range(nlayers)]
    shifts = [ _ for _ in range(nlayers)]

    hierarchies = {}
    hierarchies['RGBs'] = RGBs
    hierarchies['Ys'] = Ys
    hierarchies['Is'] = Is
    hierarchies['Qs'] = Qs

    hierarchies['features'] = features
    hierarchies['kdtrees'] = kdtrees
    hierarchies['distances'] = distances
    hierarchies['shifts'] = shifts

    CreateHierarchy(parameters, hierarchies)

    #visualization.VisualizeClassicalSR(parameters, hierarchies, 21, 21)

    #visualization.VisualizeExamplarSR(parameters, hierarchies, 20, 20)

    # go from RGB to YIQ and back
    Ys[0] = skimage.transform.resize(Ys[0], (ceil(scale * Ys[0].shape[0]), ceil(scale * Ys[0].shape[1])), order=0)
    Is[0] = skimage.transform.resize(Is[0], (ceil(scale * Is[0].shape[0]), ceil(scale * Is[0].shape[1])), order=0)
    Qs[0] = skimage.transform.resize(Qs[0], (ceil(scale * Qs[0].shape[0]), ceil(scale * Qs[0].shape[1])), order=0)
    print np.unique(Is[1] - Is[0])
    RGB_check = YIQ2RGB(Ys[0], Is[0], Qs[0])
    dataIO.WriteImage('test.png', RGB_check)

    SuperResolution(parameters, hierarchies, 1)
    #SuperResolution(parameters, hierarchies, 2)
    #SuperResolution(parameters, hierarchies, 3)
    #SuperResolution(parameters, hierarchies, 4)
    #SuperResolution(parameters, hierarchies, 5)
    #SuperResolution(parameters, hierarchies, 6)

def NearestNeighborInterpolation(parameters):
    # get useful parameters
    root_filename = parameters['root_filename']
    n = parameters['n']                             # number of layers to increase
    scale = parameters['scale']                     # scale of each layer

    # get the original image
    image_filename = 'pictures/{}.png'.format(root_filename)
    image = dataIO.ReadImage(image_filename)
    yres, xres, depth = image.shape

    magnification = pow(scale, n)
    high_yres, high_xres = (ceil(magnification * yres), ceil(magnification * xres))
    highres_image = skimage.transform.resize(image, (high_yres, high_xres, depth), order=0, mode='reflect')

    nearest_filename = 'baseline/{}-{}-nearest-neighbor.png'.format(root_filename, n)
    dataIO.WriteImage(nearest_filename, highres_image)

def BicubicInterpolation(parameters):
    # get useful parameters
    root_filename = parameters['root_filename']
    n = parameters['n']                             # number of layers to increase
    scale = parameters['scale']                     # scale of each layer

    # get the original image
    image_filename = 'pictures/{}.png'.format(root_filename)
    image = dataIO.ReadImage(image_filename)
    yres, xres, depth = image.shape

    magnification = pow(scale, n)
    high_yres, high_xres = (ceil(magnification * yres), ceil(magnification * xres))
    highres_image = skimage.transform.resize(image, (high_yres, high_xres, depth), order=3, mode='reflect')

    bicubic_filename = 'baseline/{}-{}-bicubic.png'.format(root_filename, n)
    dataIO.WriteImage(bicubic_filename, highres_image)
