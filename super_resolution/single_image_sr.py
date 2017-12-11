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

folder = '/home/bmatejek/harvard/classes/cs283/final/figures'

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

    sigma = math.sqrt(variance) / 2.5
    
    # get the radius that ends after truncate deviations
    radius = int(truncate * sigma + 0.5)

    # create a mesh grid of size (2 * radius + 1)^2
    span = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(span, span)

    # create the kernel
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)) 
    kernel = kernel / np.sum(kernel)

    return kernel

# classical constraints
class ClassicalConstraints:
    def __init__(self, iy, ix, yoffset, xoffset, level, value, weight, gaussian_sum):
        self.iy = iy
        self.ix = ix
        self.yoffset = yoffset
        self.xoffset = xoffset
        self.level = level
        self.value = value
        self.weight = weight
        self.gaussian_sum = gaussian_sum

# example based constraints
class ExemplarBasedConstraints:
    def __init__(self, iy, ix, value, weight):
        self.iy = iy
        self.ix = ix
        self.value = value
        self.weight = weight

# get the classical constraints
def GetClassicalConstraints(parameters, hierarchies, n):
    # get useful parameters
    k = parameters['k']
    scale = parameters['scale']
    diameter = parameters['diameter']
    truncate = parameters['truncate']
    min_weight = parameters['min_weight']
    
    Ys = hierarchies['Ys']
    kdtrees = hierarchies['kdtrees']
    distances = hierarchies['distances']
    features = hierarchies['features']

    # get parameters of the high resolution image
    yres, xres = Ys[0].shape
    magnification = pow(scale, n)
    high_yres, high_xres = (ceil(magnification * yres), ceil(magnification * xres))

    # get all of the gaussian blurs
    blurs = [Gaussian(pow(scale, iv), truncate) for iv in range(n + 1)]

    classical_constraints = []

    # go through every lower level
    nnonzero_entries = 0
    for level in range(n):
        mid_image = Ys[level]
        mid_yres, mid_xres = mid_image.shape

        # get the magnification from the mid level to the high level
        magnification = pow(scale, n - level)
        
        # get the range of possible misalignments (subtract this value)
        possible_alignments = [iv / float(magnification) for iv in range(-floor(scale - 0.01), floor(scale - 0.01) + 1)]
        possible_shifts = {}
        
        for ij, yshift in enumerate(possible_alignments):
            for ii, xshift in enumerate(possible_alignments):
                possible_shifts[(ij, ii)] = RemoveDC(scipy.ndimage.interpolation.shift(features[0], (yshift, xshift), order=3))

        # iterate through every pixel in this midres image
        for iy in range(mid_yres):
            for ix in range(mid_xres):
                # extract the feature at this location
                feature = ExtractFeature(features[level], iy, ix, diameter)
                max_distance = distances[level][iy,ix]

                # find the k nearest neigbhors (add one since the first neighbor is trivial)
                values, locations = kdtrees[level].query(feature, k=k+1, distance_upper_bound=max_distance)

                for value, location in zip(values, locations):
                    if value == float('inf'): continue
                    midy, midx = IndexToIndices(location, mid_xres)
                    
                    # keep track of the best offset
                    best_score = value
                    yoffset = 0
                    xoffset = 0

                    # go through all possible alignments
                    for ij in range(len(possible_alignments)):
                        for ii in range(len(possible_alignments)):
                            misaligned_feature = ExtractFeature(possible_shifts[(ij, ii)], midy, midx, diameter)
                            ssd_score = math.sqrt(np.sum(np.multiply(misaligned_feature - feature, misaligned_feature - feature)))

                            if ssd_score < best_score:
                                yoffset = ij - len(possible_alignments) / 2
                                xoffset = ii - len(possible_alignments) / 2
                                best_score = ssd_score                        

                    # get the weight of this patch depending on how good it is
                    weight = (max_distance - best_score) / max_distance
                    if weight < min_weight: continue
                    
                    # get this location to determine the number of nonzero entries
                    highy, highx = (int(round(magnification * iy)) + yoffset, int(round(magnification * ix)) + xoffset)
                    blur = blurs[n - level]
                    radius = blur.shape[0] / 2
                    gaussian_sum = 0.0
                    for ij, iv in enumerate(range(highy - radius, highy + radius + 1)):
                        for ii, iu in enumerate(range(highx - radius, highx + radius + 1)):
                            if iv < 0 or iu < 0: continue
                            if iv > high_yres - 1 or iu > high_xres - 1: continue
                            gaussian_sum += blur[ij,ii]
                            nnonzero_entries += 1

                    # only consider locations with true offsets
                    if value and not yoffset and not xoffset: continue

                    # add this constraint
                    classical_constraints.append(ClassicalConstraints(iy, ix, yoffset, xoffset, level, mid_image[midy,midx], weight, gaussian_sum))

    return classical_constraints, nnonzero_entries

def GetExampleConstraints(parameters, hierarchies, n):
    # get useful parameters
    scale = parameters['scale']
    diameter = parameters['diameter']
    truncate = parameters['truncate'] 
    min_weight = parameters['min_weight']
   
    Ys = hierarchies['Ys']
    kdtrees = hierarchies['kdtrees']
    distances = hierarchies['distances']
    features = hierarchies['features']

    # get parameters of the high resolution image
    yres, xres = Ys[0].shape
    magnification = pow(scale, n)
    high_yres, high_xres = (ceil(magnification * yres), ceil(magnification * xres))

    example_constraints = []

    blurs = [Gaussian(pow(scale, iv), 50.0) for iv in range(n)]

    # go through all levels from 0 to (n - 1)
    for level in range(n):
        mid_image = Ys[level]
        mid_yres, mid_xres = mid_image.shape

        # get the magnification from the mid level to the high level
        magnification = pow(scale, n - level)

        # iterate through every pixel in this midres image
        for iy in range(mid_yres):
            for ix in range(mid_xres):
                # extract the feature at this location
                feature = ExtractFeature(features[level], iy, ix, diameter)
                max_distance = distances[level][iy,ix]

                # look at the kdtree in the low resolution image
                value, location = kdtrees[level - n].query(feature, k=1, distance_upper_bound=max_distance)
                if value == float('inf'): continue
                lowy, lowx = IndexToIndices(location, features[level - n].shape[1])

                # where does this patch occur in the middle resolution image
                midy, midx = (int(round(magnification * lowy)), int(round(magnification * lowx)))

                # where does (iy, ix) occur in the high resolution image
                highy, highx = (int(round(magnification * iy)), int(round(magnification * ix)))

                # the high resolution diameter
                high_diameter = floor(magnification * diameter)
                high_radius = high_diameter / 2

                # get the weight of this patch depending on how good it is
                weight = (max_distance - value) / max_distance
                if weight < min_weight: continue
                
                # create constraints for the variables
                for ij, iv in enumerate(range(-high_radius, high_radius + 1)):
                    for ii, iu in enumerate(range(-high_radius, high_radius + 1)):
                        if midy + iv < 0 or midx + iu < 0: continue
                        if midy + iv > mid_yres - 1 or midx + iu > mid_xres - 1: continue
                        if highy + iv < 0 or highx + iu < 0: continue
                        if highy + iv > high_yres - 1 or highx + iu > high_xres - 1: continue

                        mid_value = mid_image[midy+iv,midx+iu]
                        
                        example_constraints.append(ExemplarBasedConstraints(highy+iv, highx+iu, mid_value, weight))

    return example_constraints, len(example_constraints)

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

    # get all of the gaussian blurs
    blurs = [Gaussian(pow(scale, iv), truncate) for iv in range(n + 1)]

    # use example based or classical only
    classical_only = False
    example_only = False
    assert (not example_only or not classical_only)

    if example_only:
        classical_constraints = []
        nnonzero_entries_from_classical = 0
    else:    
        classical_constraints, nnonzero_entries_from_classical = GetClassicalConstraints(parameters, hierarchies, n)
    
    if classical_only:
        example_constraints = []
        nnonzero_entries_from_example = 0        
    else: 
        example_constraints, nnonzero_entries_from_example = GetExampleConstraints(parameters, hierarchies, n)

    # solve the system of linear equations
    nconstraints = len(classical_constraints) + len(example_constraints)
    nnonzero_entries = nnonzero_entries_from_classical + nnonzero_entries_from_example
    data = np.zeros(nnonzero_entries, dtype=np.float32)
    i = np.zeros(nnonzero_entries, dtype=np.float32)
    j = np.zeros(nnonzero_entries, dtype=np.float32)
    b = np.zeros(nconstraints, dtype=np.float32)

    classical_weight = 1.0
    example_weight = 1.0

    nentries = 0
    for ie, constraint in enumerate(classical_constraints):
        # get the index in the lower resolution image
        iy, ix = (constraint.iy, constraint.ix)
        yoffset, xoffset = (constraint.yoffset, constraint.xoffset)
        level = constraint.level
        weight = constraint.weight
        gaussian_sum = constraint.gaussian_sum

        # find the location in the high resolution image
        magnification = pow(scale, n - level)
        highy, highx = (int(round(magnification * iy)) + yoffset, int(round(magnification * ix)) + xoffset)

        # get the gaussian blur around this location
        blur = blurs[n - level]
        radius = blur.shape[0] / 2

        for ij, iv in enumerate(range(highy - radius, highy + radius + 1)):
            for ii, iu in enumerate(range(highx - radius, highx + radius + 1)):
                if iv < 0 or iu < 0: continue
                if iv > high_yres - 1 or iu > high_xres - 1: continue

                # get this index
                high_index = IndicesToIndex(iv, iu, high_xres)
                i[nentries] = ie
                j[nentries] = high_index
                data[nentries] = classical_weight * weight * blur[ij,ii] / gaussian_sum
                nentries += 1

        # set the value of the system
        b[ie] = classical_weight * weight * constraint.value

    nclassical_constraints = len(classical_constraints)
    for ie, constraint in enumerate(example_constraints):
        iy, ix = (constraint.iy, constraint.ix)
        index = IndicesToIndex(iy, ix, high_xres)
        weight = constraint.weight

        i[nentries] = ie + nclassical_constraints
        j[nentries] = index
        data[nentries] = example_weight * weight

        # set the value of the system
        b[ie + nclassical_constraints] = example_weight * weight * constraint.value
        nentries += 1

    # create the sparse matrix
    A = scipy.sparse.coo_matrix((data, (i, j)), shape=(nconstraints, high_yres * high_xres))

    # give an inital estimage (use a bicubic upsampled blured image)
    if parameters['initialize']:
        x0 = skimage.transform.resize(Ys[0], (high_yres, high_xres), order=3, mode='reflect').flatten()
        r0 = b - (A.tocsr() * x0)
        dx, _, _, _,_, _, _, _, _, _  = scipy.sparse.linalg.lsqr(A, r0, show=True, iter_lim=100000)
        x = x0 + dx
    else:
        x, _, _, _, _, _, _, _, _, _ = scipy.sparse.linalg.lsqr(A, b, show=True, iter_lim=100000)

    smally, smallx = IndexToIndices(int(round(np.amin(j))), high_xres)
    largey, largex = IndexToIndices(int(round(np.amax(j))), high_xres)
    print 'Original minimum x: {}'.format(np.amin(x))
    print 'Original maximum x: {}'.format(np.amax(x))
    x[x < 0] = 0
    x[x > 1] = 1
    maximum = np.amax(x)
    minimum = np.amin(x)
    
    print 'Minimum Index: ({}, {})'.format(smally, smallx)
    print 'Maximum Index: ({}, {})'.format(largey, largex)
    print 'Minimum Data: {}'.format(np.amin(data))
    print 'Maximum Data: {}'.format(np.amax(data))
    print 'Minumum B: {}'.format(np.amin(b))
    print 'Maximum B: {}'.format(np.amax(b))
    print 'Minimum x: {}'.format(np.amin(x))
    print 'Maximum x: {}'.format(np.amax(x))    

    highres_Y = np.zeros((high_yres, high_xres), dtype=np.float32)
    for iy in range(high_yres):
        for ix in range(high_xres):
            index = IndicesToIndex(iy, ix, high_xres)
            highres_Y[iy,ix] = (x[index] - minimum) / (maximum - minimum)

    low_mean = np.mean(Ys[0])
    high_mean = np.mean(highres_Y)
    low_std = np.std(Ys[0])
    high_std = np.std(highres_Y)

    # match the distributions
    for iy in range(high_yres):
        for ix in range(high_xres):
            highres_Y[iy,ix] = (highres_Y[iy,ix] - high_mean) / high_std * low_std + low_mean

    Ys[n] = highres_Y

    if classical_only:
        grayscale_filename = '{}/results/{}-super-resolution-grayscale-classical-{}.png'.format(folder, root_filename, n)
        rgb_filename = '{}/results/{}-super-resolution-RGB-classical-{}.png'.format(folder, root_filename, n)
    elif example_only:
        grayscale_filename = '{}/results/{}-super-resolution-grayscale-example-{}.png'.format(folder, root_filename, n)
        rgb_filename = '{}/results/{}-super-resolution-RGB-example-{}.png'.format(folder, root_filename, n)
    else:
        grayscale_filename = '{}/results/{}-super-resolution-grayscale-{}.png'.format(folder, root_filename, n)
        rgb_filename = '{}/results/{}-super-resolution-RGB-{}.png'.format(folder, root_filename, n)
    
    dataIO.WriteImage(grayscale_filename, Ys[n])
    RGBs[n] = YIQ2RGB(Ys[n], Is[n], Qs[n])
    dataIO.WriteImage(rgb_filename, RGBs[n])


    features[n] = RemoveDC(Ys[n])
    kdtrees[n] = CreateKDTree(features[n], diameter)
    distances[n], shifts[n] = GoodDistance(features[n], diameter)


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

    # find similar patches at various scales
    visualization.VisualizeRedundancy(parameters, hierarchies)
    
    #visualization.VisualizeClassicalSR(parameters, hierarchies, 21, 21)

    #visualization.VisualizeExamplarSR(parameters, hierarchies, 20, 20)

    SuperResolution(parameters, hierarchies, 1)
    SuperResolution(parameters, hierarchies, 2)
    SuperResolution(parameters, hierarchies, 3)
    SuperResolution(parameters, hierarchies, 4)
    # SuperResolution(parameters, hierarchies, 5)
    # SuperResolution(parameters, hierarchies, 6)

def NearestNeighborInterpolation(parameters):
    # get useful parameters
    root_filename = parameters['root_filename']
    n = parameters['n']                             # number of layers to increase
    scale = parameters['scale']                     # scale of each layer

    # get the original image
    image_filename = 'pictures/{}.png'.format(root_filename)
    image = dataIO.ReadImage(image_filename)
    yres, xres, depth = image.shape

    for iv in range(1, n + 1):
        magnification = pow(scale, iv)
        high_yres, high_xres = (ceil(magnification * yres), ceil(magnification * xres))
        highres_image = skimage.transform.resize(image, (high_yres, high_xres, depth), order=0, mode='reflect')

        nearest_filename = '{}/baseline/{}-{}-nearest-neighbor.png'.format(folder, root_filename, iv)
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

    for iv in range(1, n + 1):
        magnification = pow(scale, iv)
        high_yres, high_xres = (ceil(magnification * yres), ceil(magnification * xres))
        highres_image = skimage.transform.resize(image, (high_yres, high_xres, depth), order=3, mode='reflect')

        bicubic_filename = '{}/baseline/{}-{}-bicubic.png'.format(folder, root_filename, iv)
        dataIO.WriteImage(bicubic_filename, highres_image)
