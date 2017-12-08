import math
import struct
import numpy as np
import scipy.ndimage
import skimage.transform
import os.path

from scipy.spatial import KDTree
from numba import jit

from ibex.utilities import dataIO

# for KDTree to work
import sys
sys.setrecursionlimit(100000)

# define conveniet ceil and floor functions
def ceil(value): return int(math.ceil(value))
def floor(value): return int(math.floor(value))

# conversion between linear and quadric spaces
def IndicesToIndex(iy, ix, xres): return iy * xres + ix
def IndexToIndices(index, xres): return (index / xres, index % xres)

# convert the RGB image to grayscale
def RGB2Gray(image):
    yres, xres, _ = image.shape
    grayscale = 0.2126 * image[:,:,2] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,0]
    return grayscale

# remove average grayscale from image
def RemoveDC(grayscale):
    return grayscale - np.mean(grayscale)

# create a gaussian kernel
def SSDGaussian(diameter):
    sys.stderr.write('Warning: using diameter of ({}, {}) for SSD'.format(diameter[0], diameter[1]))
    # no variance equals the delta function
    sigma = diameter[0] / 6.4
    
    # get the radius that ends after truncate deviations
    radius = (diameter[0] / 2, diameter[1] / 2)

    # create a mesh grid of size (2 * radius + 1)^2
    xspan = np.arange(-radius[0], radius[0] + 1)
    yspan = np.arange(-radius[1], radius[1] + 1)
    xx, yy = np.meshgrid(xspan, yspan)

    # create the kernel
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)) 
    kernel = kernel / np.sum(kernel)

    return kernel.flatten()

# call once globally to save time
ssd_kernel = SSDGaussian((5, 5))

# extract the feature from this location
def ExtractFeature(grayscale, iy, ix, diameter):
    # get convenient variables
    radius = (diameter[0] / 2, diameter[1] / 2)
    yres, xres = grayscale.shape

    # see if reflection is needed
    if iy > radius[0] - 1 and ix > radius[1] - 1 and iy < yres - radius[0] and ix < xres - radius[1]:
        return np.multiply(ssd_kernel, grayscale[iy-radius[0]:iy+radius[0]+1,ix-radius[1]:ix+radius[1]+1].flatten())
    else: 
        return sys.maxint * np.ones(diameter, dtype=np.float32).flatten()
    
# create the kdtree of features for this image
def CreateKDTree(grayscale, diameter):
    # get useful parameters
    yres, xres = grayscale.shape
    nfeatures = yres * xres
    feature_size = diameter[0] * diameter[1]
    features = np.zeros((nfeatures, feature_size), dtype=np.float32)
    radius = (diameter[0] / 2, diameter[1] / 2)
    for iy in range(yres):
        for ix in range(xres):
            features[IndicesToIndex(iy, ix, xres),:] = ExtractFeature(grayscale, iy, ix, diameter)

    return KDTree(features)

# get the patch-specific good distances
def GoodDistance(grayscale, diameter):
    # get useful parameters
    yres, xres = grayscale.shape
    # shift the image by half a pixel
    shift = scipy.ndimage.interpolation.shift(grayscale, (0.5, 0.5))
    distance = np.zeros((yres, xres), dtype=np.float32)

    for iy in range(yres):
        for ix in range(xres):
            grayscale_feature = ExtractFeature(grayscale, iy, ix, diameter)
            shift_feature = ExtractFeature(shift, iy, ix, diameter)

            distance[iy,ix] = math.sqrt(np.sum(np.multiply(grayscale_feature - shift_feature, grayscale_feature - shift_feature)))

    return distance

# create a hierarchy of images, grayscales, and kdtrees
def CreateHierarchy(parameters, hierachies):
    # get useful parameters
    root_filename = parameters['root_filename']
    m = parameters['m']                             # number of down level supports
    scale = parameters['scale']                     # scale to increase each layer
    truncate = parameters['truncate']               # number of sigma in gaussian blur
    diameter = parameters['diameter']               # diameter of features

    images = hierachies['images']
    grayscales = hierachies['grayscales']
    kdtrees = hierachies['kdtrees']
    distances = hierachies['distances']

    # read in the first image and convert to grayscale
    filename = 'pictures/{}.png'.format(root_filename)
    images[0] = dataIO.ReadImage(filename)
    grayscales[0] = RGB2Gray(images[0])
    yres, xres = grayscales[0].shape

    # output the input image into the hierarchy subfolder
    grayscale_filename = 'hierarchy/{}-grayscale-0.png'.format(root_filename)
    image_filename = 'hierarchy/{}-image-0.png'.format(root_filename)
    dataIO.WriteImage(grayscale_filename, grayscales[0])
    dataIO.WriteImage(image_filename, images[0])

    # go through all levels of the hierarchy
    for iv in range(-1, -(m + 1), -1):
        # get the standard deviation of the blur filter
        sigma = math.sqrt(pow(scale, -iv))

        # filter the input image 
        blurred_grayscale = scipy.ndimage.filters.gaussian_filter(grayscales[0], sigma, truncate=truncate)
        blurred_red = scipy.ndimage.filters.gaussian_filter(images[0][:,:,0], sigma, truncate=truncate)
        blurred_green = scipy.ndimage.filters.gaussian_filter(images[0][:,:,1], sigma, truncate=truncate)
        blurred_blue = scipy.ndimage.filters.gaussian_filter(images[0][:,:,2], sigma, truncate=truncate)

        # get the scale difference between this level and the first level
        magnification = pow(scale, iv)
        low_yres, low_xres = (ceil(yres * magnification), ceil(xres * magnification))
        
        # perform bilinear interpolation
        grayscales[iv] = skimage.transform.resize(blurred_grayscale, (low_yres, low_xres), order=1, mode='reflect')
        images[iv] = np.zeros((low_yres, low_xres, 3), dtype=np.float32)
        images[iv][:,:,0] = skimage.transform.resize(blurred_red, (low_yres, low_xres), order=1, mode='reflect')
        images[iv][:,:,1] = skimage.transform.resize(blurred_green, (low_yres, low_xres), order=1, mode='reflect')
        images[iv][:,:,2] = skimage.transform.resize(blurred_blue, (low_yres, low_xres), order=1, mode='reflect')

        # output the images
        grayscale_filename = 'hierarchy/{}-grayscale-{}.png'.format(root_filename, -iv)
        image_filename = 'hierarchy/{}-image-{}.png'.format(root_filename, -iv)
        dataIO.WriteImage(grayscale_filename, grayscales[iv])
        dataIO.WriteImage(image_filename, images[iv])

    # remove all DC components from grayscale images for feature extraction
    for iv in range(0, -(m + 1), -1):
        grayscales[iv] = RemoveDC(grayscales[iv])
        kdtrees[iv] = CreateKDTree(grayscales[iv], diameter)
        distances[iv] = GoodDistance(grayscales[iv], diameter)

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

class Constraint:
    def __init__(self, iy, ix, l, value):
        self.iy = iy
        self.ix = ix
        self.l = l
        self.value = value

# save the constraints file
def SaveConstraints(constraints_filename, constraints):
    # save the constraints to the cache
    with open(constraints_filename, 'wb') as fd:
        fd.write(struct.pack('i', len(constraints)))
        for constraint in constraints:
            fd.write(struct.pack('iiif', constraint.iy, constraint.ix, constraint.l, constraint.value))

# read the constraints file
def ReadConstraints(constraints_filename):
    constraints = []
    with open(constraints_filename, 'rb') as fd:
        nconstraints, = struct.unpack('i', fd.read(4))
        for iv in range(nconstraints):
            iy, ix, l, value, = struct.unpack('iiif', fd.read(16))
            constraints.append(Constraint(iy, ix, l, value))

    return constraints

# increase the resolution to level in one pass
def SinglePassSuperResolution(parameters, hierachies, n):
    # get useful parameters
    root_filename = parameters['root_filename']
    n = parameters['n']
    m = parameters['m']
    k = parameters['k']
    scale = parameters['scale']
    diameter = parameters['diameter']
    truncate = parameters['truncate']

    grayscales = hierachies['grayscales']
    kdtrees = hierachies['kdtrees']
    distances = hierachies['distances']

    L = grayscales[0]
    Ldist = distances[0]
    yres, xres = L.shape
    radius = (diameter[0] / 2, diameter[1] / 2)

    lowres_kdtree = kdtrees[-1]
    lowres_grayscale = grayscales[-1]
    magnification = pow(scale, n)

    high_yres, high_xres = (ceil(yres * magnification), ceil(xres * magnification))
    highres_image = np.zeros((high_yres, high_xres), dtype=np.float32)

    constraints = []
    constraints_filename = 'cache/{}-single-pass-constraints.cache'.format(root_filename)

    if os.path.exists(constraints_filename):
        constraints = ReadConstraints(constraints_filename)
    else:
        for iy in range(radius[0], yres - radius[0]):
            for ix in range(radius[1], xres - radius[1]):
                print '{} {}'.format(iy, ix)
                sys.stdout.flush()

                # create one constraint for this level
                constraints.append(Constraint(iy, ix, 0, L[iy,ix]))

                # get the feature at this location
                feature = ExtractFeature(L, iy, ix, diameter)

                # find the closest feature in the lowres image
                values, locations = lowres_kdtree.query(feature, 3, distance_upper_bound=Ldist[iy,ix])
                for value, location in zip(values, locations):
                    if value == float('inf'): continue

                    # get the low resolution location
                    lowy, lowx = IndexToIndices(location, lowres_grayscale.shape[1])
                    # get the corresponding location and window in L
                    Ly, Lx = (int(round(lowy * magnification)), int(round(lowx * magnification)))
                    Lradius = (floor(diameter[0] * magnification) / 2, floor(diameter[1] * magnification) / 2)
                    if Ly - Lradius[0] < 0 or Lx - Lradius[1] < 0: continue
                    if Ly + Lradius[0] > yres - 1 or Lx + Lradius[1] > xres - 1: continue            
                    Lwindow = L[Ly-Lradius[0]:Ly+Lradius[0]+1,Lx-Lradius[1]:Lx+Lradius[1]+1]

                    # get the high res location
                    highy, highx = (int(round(iy * magnification)), int(round(ix * magnification)))
                    if highy - Lradius[0] < 0 or highx - Lradius[1] < 0: continue
                    if highy + Lradius[0] > high_yres - 1 or highx + Lradius[1] > high_xres + 1: continue

                    # create a constraint for every pixel at this level
                    for ij, iv in enumerate(range(highy - Lradius[0], highy + Lradius[0] + 1)):
                        for ii, iu in enumerate(range(highx - Lradius[1], highx + Lradius[1] + 1)):
                            constraints.append(Constraint(iy, ix, 1, Lwindow[ij,ii]))

                    highres_image[highy-Lradius[0]:highy+Lradius[0]+1,highx-Lradius[1]:highx+Lradius[1]+1] = Lwindow
        SaveConstraints(constraints_filename, constraints)
        dataIO.WriteImage('output-original.png', highres_image)
    # create gaussian blurs for every difference
    blurs = [Gaussian(pow(scale, iv), truncate) for iv in range(n + 1)]

    index = 0
    nconstraints = 0
    for ie, constraint in enumerate(constraints):
        iy = constraint.iy
        ix = constraint.ix
        l = constraint.l

        # if the level is the desired level
        if l == n:
            index += 1
        else:
            
            # get the high resolution location
            magnification = pow(scale, n - l)
            highy, highx = (int(round(iy * magnification)), int(round(ix * magnification)))

            # take a gaussian blur around the location in the low res image
            gaussian_blur = blurs[n - l]
            gaussian_radius = (gaussian_blur.shape[0] / 2, gaussian_blur.shape[1] / 2)

            if highy - gaussian_radius[0] < 0 or highx - gaussian_radius[1] < 0: continue
            if highy + gaussian_radius[0] > yres - 1 or highx + gaussian_radius[1] > xres - 1: continue

            for ij, iv in enumerate(range(highy - gaussian_radius[0], highy + gaussian_radius[0] + 1)):
                for ii, iu in enumerate(range(highx - gaussian_radius[1], highx + gaussian_radius[1] + 1)):
                    if iv < 0 or iv > high_yres - 1: continue
                    if iu < 0 or iu > high_xres - 1: continue
                    index += 1
        nconstraints += 1

    # create a system of sparse equations
    #nconstraints = len(constraints)
    nnonzero = index

    data = np.zeros(nnonzero, dtype=np.float32)
    i = np.zeros(nnonzero, dtype=np.float32)
    j = np.zeros(nnonzero, dtype=np.float32)
    
    b = np.zeros(nconstraints, dtype=np.float32)

    index = 0
    index2 = 0
    for ie, constraint in enumerate(constraints):
        iy = constraint.iy
        ix = constraint.ix
        l = constraint.l
        value = constraint.value


        # if the level is the desired level
        if l == n:
            high_index = IndicesToIndex(iy, ix, highres_image.shape[1])

            i[index] = index2
            j[index] = high_index
            data[index] = 1
            index += 1
        else:
            
            # get the high resolution location
            magnification = pow(scale, n - l)
            highy, highx = (int(round(iy * magnification)), int(round(ix * magnification)))

            # take a gaussian blur around the location in the low res image
            gaussian_blur = blurs[n - l]
            gaussian_radius = (gaussian_blur.shape[0] / 2, gaussian_blur.shape[1] / 2)

            if highy - gaussian_radius[0] < 0 or highx - gaussian_radius[1] < 0: continue
            if highy + gaussian_radius[0] > yres - 1 or highx + gaussian_radius[1] > xres - 1: continue

            for ij, iv in enumerate(range(highy - gaussian_radius[0], highy + gaussian_radius[0] + 1)):
                for ii, iu in enumerate(range(highx - gaussian_radius[1], highx + gaussian_radius[1] + 1)):
                    if iv < 0 or iv > high_yres - 1: continue
                    if iu < 0 or iu > high_xres - 1: continue

                    # get the index
                    high_index = IndicesToIndex(iv, iu, high_xres)

                    i[index] = ie
                    j[index] = high_index
                    data[index] = gaussian_blur[ij,ii]
                    index += 1

        # set the least square values
        b[index2] = value
        index2 += 1
    
    print nconstraints
    print nnonzero
    print b.size
    print np.amax(data)
    print np.amin(data)
    print np.amax(i)
    print np.amax(j)
    print np.amin(i)
    print np.amin(j)
    print high_yres
    print high_xres
    sparse_matrix = scipy.sparse.coo_matrix((data, (i, j)), shape=(nconstraints, high_yres * high_xres))
    H, _, _, _,_, _, _, _, _, _ = scipy.sparse.linalg.lsqr(sparse_matrix, b, show=True)
    
    maximum = np.amax(H)
    minimum = np.amin(H)
    print 'Minimum: {}'.format(minimum)
    print 'Maximum: {}'.format(maximum)

    for iy in range(high_yres):
        for ix in range(high_xres):
            index = IndicesToIndex(iy, ix, high_xres)

            highres_image[iy,ix] = (H[index] - minimum) / (maximum - minimum)

    grayscales[n] = highres_image
    dataIO.WriteImage('output.png', grayscales[n])


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
    images = [ _ for _ in range(nlayers)]
    grayscales = [ _ for _ in range(nlayers)]
    kdtrees = [ _ for _ in range(nlayers)]
    distances = [ _ for _ in range(nlayers)]


    hierachies = {}
    hierachies['images'] = images
    hierachies['grayscales'] = grayscales
    hierachies['kdtrees'] = kdtrees
    hierachies['distances'] = distances

    CreateHierarchy(parameters, hierachies)

    # diameter = parameters['diameter']
    # radius = (diameter[0] / 2, diameter[1] / 2)
    # distance = distances[0]
    # grayscale = grayscales[0]
    # k = 10
    # nvalids = [0 for iv in range(k + 1)]
    # yres, xres = grayscale.shape
    # for iy in range(radius[0], yres - radius[0]):
    #     for ix in range(radius[1], xres - radius[1]):
    #         # get this feature
    #         feature = ExtractFeature(grayscale, iy, ix, diameter)

    #         # go through every lower level
    #         values, locations = kdtrees[-6].query(feature, k=k, distance_upper_bound=distance[iy,ix])

    #         nvalid = 0
    #         for ie in range(len(locations)):
    #             if values[ie] < distance[iy,ix]:
    #                 nvalid += 1

    #         nvalids[nvalid] += 1
    #     print nvalids
    #     sys.stdout.flush()


    #VisualizeSelectionProcess(parameters, hierachies)

    SinglePassSuperResolution(parameters, hierachies, 1)





















# create a bounding box of size diameter around this location
def BoundingBox(image, iy, ix, diameter):
    radius = (ceil(diameter[0] / 2), ceil(diameter[1] / 2))
    yres, xres, depth = image.shape

    output_image = np.zeros((yres, xres, depth), dtype=np.float32)
    output_image[:,:,:] = image[:,:,:]
    
    lowy, lowx = (iy - (radius[0] + 1), ix - (radius[1] + 1))
    highy, highx = (iy + (radius[0] + 1), ix + (radius[1] + 1))

    output_image[lowy:highy+1,lowx,:] = 1.0
    output_image[lowy:highy+1,highx,:] = 1.0
    output_image[lowy,lowx:highx+1,:] = 1.0
    output_image[highy,lowx:highx+1,:] = 1.0

    return output_image

# show a visualization of the process for equation on p 353
def VisualizeSelectionProcess(parameters, hierachies):
    # get useful parameters
    root_filename = parameters['root_filename']    
    diameter = parameters['diameter']               # diameter of features
    scale = parameters['scale']                     # scale to increase each layer

    images = hierachies['images']
    grayscales = hierachies['grayscales']
    kdtrees = hierachies['kdtrees']

    # arbitray location for example
    iy, ix = (80, 80)

    feature = ExtractFeature(grayscales[0], iy, ix, diameter)
    highres_image = BoundingBox(images[0], iy, ix, diameter)

    # get the feature from both lower levels
    for iv in range(-1, -6, -1):
        # find the closest feature in this level
        _, location = kdtrees[iv].query(feature, 1)
        lowy, lowx = IndexToIndices(location, grayscales[iv].shape[1])

        # create the image 
        lowres_image = BoundingBox(images[iv], lowy, lowx, diameter)

        visualization_filename = 'visualizations/{}-bbox-{}.png'.format(root_filename, -iv)
        dataIO.WriteImage(visualization_filename, lowres_image)

        # get the high res locations
        magnification = pow(scale, -iv)
        highy, highx = (int(round(lowy * magnification)), int(round(lowx * magnification)))
        high_diameter = (diameter[0] * magnification, diameter[1] * magnification)

        highres_image = BoundingBox(highres_image, highy, highx, high_diameter)

    visualization_filename = 'visualizations/{}-bbox-0.png'.format(root_filename)
    dataIO.WriteImage(visualization_filename, highres_image)

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






# grayscales = hierachies['grayscales']
# kdtrees = hierachies['kdtrees']
# distances = hierachies['distances']

# # get the low resolution image
# L = grayscales[0]
# distance = distances[0]
# yres, xres = L.shape
# radius = (diameter[0] / 2, diameter[1] / 2)

# # use the cache if it exists
# constraints_filename = 'cache/{}-single-pass-constraints.cache'.format(root_filename)

# if not os.path.exists(constraints_filename):        
#     constraints = []

#     # go through every level
#     for l in range(-1, (-m + 1), -1):
#         kdtree = kdtrees[l]
#         magnification = pow(scale, -l)
#         high_radius = (ceil(diameter[0] * magnification) / 2, ceil(diameter[1] * magnification) / 2)

#         for iy in range(radius[0], yres - radius[0]):
#             for ix in range(radius[1], xres - radius[1]):
#                 print '{} {}'.format(iy, ix)
#                 sys.stdout.flush()
#                 # get the valid locations
#                 feature = ExtractFeature(L, iy, ix, diameter)
#                 value, location = kdtree.query(feature, k, distance_upper_bound=distance[iy,ix])

#                 # # iterate over all locations
#                 # for index, location in enumerate(locations):
#                 if value > distance[iy,ix]: continue

#                 lowy, lowx = IndexToIndices(location, grayscales[l].shape[1])
#                 Ly, Lx = (int(round(lowy * magnification)), int(round(lowx * magnification)))
#                 highy, highx = (int(round(Ly * magnification)), int(round(Lx * magnification)))

#                 # extract the local region
#                 Q = L[Ly-high_radius[0]:Ly+high_radius[0]+1,Lx-high_radius[1]:Lx+high_radius[1]+1]
#                 if not Q.size == (2 * high_radius[0] + 1) * (2 * high_radius[1] + 1): continue

#                 for iv, ij in enumerate(range(highy - high_radius[0], highy + high_radius[0] + 1)):
#                     for iu, ii in enumerate(range(highx - high_radius[1], highx + high_radius[1] + 1)):
#                         constraints.append(Constraint(ij, ii, -l, Q[iv,iu]))

#     SaveConstraints(constraints_filename, constraints)
# else:
#     constraints = ReadConstraints(constraints_filename)
# nconstraints = len(constraints)

# # get the size of the final image
# magnification = pow(scale, n)
# high_yres, high_xres = (ceil(magnification * yres), ceil(magnification * xres))

# H = np.zeros((nconstraints, high_yres * high_xres), dtype=np.float32)
# b = np.zeros(nconstraints)

# # create gaussian blurs for every difference
# blurs = [GaussianBlur(pow(scale, iv), truncate) for iv in range(n)]

# for iv, constraint in enumerate(constraints):
#     print '{}/{}'.format(iv, nconstraints)
#     sys.stdout.flush()
#     # get the particular of this constraint
#     iy = constraint.iy
#     ix = constraint.ix
#     l = constraint.l
#     p = constraint.value
    
#     # get the high resolution locations
#     difference = n - l
#     magnification = pow(scale, difference)
#     highy, highx = (int(round(magnification * iy)), int(round(magnification * ix)))
#     kernel = blurs[difference]
#     kernel_radius = (kernel.shape[0] / 2, kernel.shape[1], 2)

#     # iterate over the entire kernel
#     for iv, iy in enumerate(range(highy - kernel_radius[0], highy + kernel_radius[0] + 1)):
#         if iy < 0 or iy > high_yres - 1: continue
#         for iu, ix in enumerate(range(highx - kernel_radius[1], highx + kernel_radius[1] + 1)):
#             if ix < 0 or ix > high_xres - 1: continue

#             high_index = IndexToIndices(iy, ix, high_xres)
#             H[iv,high_index] = kernel[iv,iu]

#     b[iv] = constraint.value

#     # solve the linear system
#     np.linalg.solve(H, b)