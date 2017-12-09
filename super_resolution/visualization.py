import numpy as np
import skimage.transform
import scipy.ndimage

from ibex.utilities import dataIO
from ibex.super_resolution.util import *

# folder for all of the figures
folder = '/home/bmatejek/harvard/classes/cs283/final/figures'


# magnify the image and return (nearest neighbor)
def MagnifyImage(image, magnification):
    if len(image.shape) == 3:
        yres, xres, depth = image.shape
        high_yres, high_xres = (ceil(magnification * yres), ceil(magnification * xres))
        return skimage.transform.resize(image, (high_yres, high_xres, depth), order=0, mode='reflect')
    else:
        yres, xres = image.shape
        high_yres, high_xres = (ceil(magnification * yres), ceil(magnification * xres))
        return skimage.transform.resize(image, (high_yres, high_xres), order=0, mode='reflect')

# create a bounding box of size diameter around this location
def DrawBoundingBox(image, iy, ix, diameter):
    radius = diameter / 2
    yres, xres, depth = image.shape

    output_image = np.zeros((yres, xres, depth), dtype=np.float32)
    output_image[:,:,:] = image[:,:,:]
    
    lowy, lowx = (iy - (radius + 1), ix - (radius + 1))
    highy, highx = (iy + (radius + 1), ix + (radius + 1))

    output_image[lowy:highy+1,lowx,:] = 1.0
    output_image[lowy:highy+1,highx,:] = 1.0
    output_image[lowy,lowx:highx+1,:] = 1.0
    output_image[highy,lowx:highx+1,:] = 1.0

    return output_image

# visualize the classical SR features
def VisualizeClassicalSR(parameters, hierarchies, iy, ix):
    # get useful parameters
    root_filename = parameters['root_filename']
    diameter = parameters['diameter']               # diameter of feature
    k = parameters['k']                             # number of parameters at scale

    RGBs = hierarchies['RGBs']
    Ys = hierarchies['Ys']
    features = hierarchies['features']
    kdtrees = hierarchies['kdtrees']
    distances = hierarchies['distances']
    shifts = hierarchies['shifts']

    # useful variables for this function
    yres, xres = Ys[0].shape
    iy, ix = (21, 21)
    enlargement = 80
    radius = diameter / 2
    max_distance = distances[0][iy,ix]

    # save the gaussian kernel
    magnified_kernel = MagnifyImage(ssd_kernel.reshape(diameter, diameter), enlargement)
    kernel_filename = '{}/features/kernel.png'.format(folder)
    dataIO.WriteImage(kernel_filename, magnified_kernel)

    # get the feature and magnify
    feature = ExtractFeature(features[0], iy, ix, diameter, weighted=False).reshape((diameter, diameter))
    feature_filename = '{}/features/{}-feature-({}-{}).png'.format(folder, root_filename, iy, ix)
    dataIO.WriteImage(feature_filename, MagnifyImage(feature, enlargement))

    # shift the feature by 0.5 pixels and display the results
    shift_feature = ExtractFeature(shifts[0], iy, ix, diameter, weighted=False).reshape(diameter, diameter)
    shift_filename = '{}/features/{}-shift-({}-{}).png'.format(folder, root_filename, iy, ix)
    dataIO.WriteImage(shift_filename, MagnifyImage(shift_feature, enlargement))

    # white out this box
    boxed_image = DrawBoundingBox(RGBs[0], iy, ix, diameter)
    original_filename = '{}/features/{}-sample-({}-{}).png'.format(folder, root_filename, iy, ix)
    dataIO.WriteImage(original_filename, MagnifyImage(boxed_image, enlargement))

    # get the k features within scale
    SSD_filename = '{}/features/{}-SSD-scores-within.log'.format(folder, root_filename)
    with open(SSD_filename, 'w') as fd:
        fd.write('Shift Score: {}\n'.format(max_distance))
        values, locations = kdtrees[0].query(ExtractFeature(features[0], iy, ix, diameter), k=k, distance_upper_bound=max_distance)
        for index, (value, location) in enumerate(zip(values, locations)):
            if value == float('inf'): continue
            # get the location
            iv, iu = IndexToIndices(location, xres)
            # skip the trivial closest example
            if iv == iy and iu == ix: continue

            # save this feature location
            matched_feature = ExtractFeature(features[0], iv, iu, diameter, weighted=False).reshape(diameter, diameter)
            matched_filename = '{}/features/{}-matched_feature-{}.png'.format(folder, root_filename, index)
            dataIO.WriteImage(matched_filename, MagnifyImage(matched_feature, enlargement))

            boxed_image = DrawBoundingBox(boxed_image, iv, iu, diameter)

            # write this value
            fd.write('Match Score: {}\n'.format(value))

    # compute the sub pixel alignments
    scale = 2                                          # hardcode the scale factor for better images

    # compute each of the possible misalignments


    within_scale_filename = '{}/features/{}-within-scale-matches.png'.format(folder, root_filename)
    dataIO.WriteImage(within_scale_filename, MagnifyImage(boxed_image, enlargement))


# visualize the examplar SR features
def VisualizeExamplarSR(parameters, hierarchies, iy, ix):
    # get useful parameters
    root_filename = parameters['root_filename']
    m = parameters['m']                             # number of lower levels
    scale = parameters['scale']                     # the scaling factor
    diameter = parameters['diameter']               # diameter of feature

    RGBs = hierarchies['RGBs']
    Ys = hierarchies['Ys']
    features = hierarchies['features']
    kdtrees = hierarchies['kdtrees']
    distances = hierarchies['distances']

    # useful variables for this function
    yres, xres = Ys[0].shape
    enlargement = 80
    radius = diameter / 2
    max_distance = distances[0][iy,ix] 

    # get the feature and magnify
    feature = ExtractFeature(features[0], iy, ix, diameter, weighted=False).reshape((diameter, diameter))
    feature_filename = '{}/features/{}-feature-({}-{}).png'.format(folder, root_filename, iy, ix)
    dataIO.WriteImage(feature_filename, MagnifyImage(feature, enlargement))

    # white out this box
    boxed_image = DrawBoundingBox(RGBs[0], iy, ix, diameter)
    original_filename = '{}/features/{}-sample-({}-{}).png'.format(folder, root_filename, iy, ix)
    dataIO.WriteImage(original_filename, MagnifyImage(boxed_image, enlargement))

    # outline bounding box
    boxed_image = DrawBoundingBox(RGBs[0], iy, ix, diameter)

    SSD_filename = '{}/features/{}-SSD-scores-across.log'.format(folder, root_filename)
    with open(SSD_filename, 'w') as fd:
        fd.write('Shift Score: {}\n'.format(max_distance))
        # iterate over all levels and find the closest locations
        for level in range(-1, -(m + 1), -1):
            # find the nearest feature at each level
            value, location = kdtrees[level].query(ExtractFeature(features[0], iy, ix, diameter), k=1, distance_upper_bound=max_distance)
            if value == float('inf'): continue
            # get the location in the downsampled image
            iv, iu = IndexToIndices(location, Ys[level].shape[1])

            matched_feature = ExtractFeature(features[level], iv, iu, diameter, weighted=False).reshape(diameter, diameter)
            matched_filename = '{}/features/{}-matched_feature-level{}.png'.format(folder, root_filename, level)
            dataIO.WriteImage(matched_filename, MagnifyImage(matched_feature, enlargement))

            # find the location in the higher resolution image
            magnification = pow(scale, -level)
            highv, highu = (int(round(magnification * iv)), int(round(magnification * iu)))
            # get the diameter of the higher resolution location
            high_diameter = floor(magnification * diameter)
            high_radius = high_diameter / 2

            # get the corresponding location
            window = Ys[0][highv-high_radius:highv+high_radius+1,highu-high_radius:highu+high_radius+1]
            window_filename = '{}/features/{}-matched_feature-level{}-window.png'.format(folder, root_filename, level)
            dataIO.WriteImage(window_filename, MagnifyImage(window, enlargement))
            
            # box this location in the image
            boxed_image = DrawBoundingBox(boxed_image, highv, highu, high_diameter)
            
            fd.write('Match Score: {}\n'.format(value))

    across_scale_filename = '{}/features/{}-across-scale-matches.png'.format(folder, root_filename)
    dataIO.WriteImage(across_scale_filename, MagnifyImage(boxed_image, enlargement))