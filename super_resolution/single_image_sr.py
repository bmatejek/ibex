# import math
# import numpy as np
# import skimage.transform

# from numba import jit
# from scipy.spatial import KDTree
# import scipy.ndimage
# import scipy.signal

# from ibex.utilities import dataIO



# # small utility functions
# @jit(nopython=True)
# def ceil(value): return int(math.ceil(value))
# @jit(nopython=True)
# def floor(value): return int(math.floor(value))

# def Gaussian(diameter, sigma):
#     radius = (diameter[0] / 2, diameter[1] / 2)
#     y_range = np.arange(-radius[0], radius[0] + 1)
#     x_range = np.arange(-radius[1], radius[1] + 1)
    
#     xx, yy = np.meshgrid(x_range, y_range)
#     kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)) 
#     kernel = kernel / np.sum(kernel)

#     return kernel

# # convert the image to grayscale
# def RGB2Gray(image):
#     yres, xres, depth = image.shape
#     assert (depth == 3)
#     # G = 0.2126 R + 0.7152 G + 0.0722 B
#     grayscale = 0.2126 * image[:,:,2] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,0]
#     return grayscale

# # apply a gaussian blue
# def GaussianBlur(image, sigma):
#     return scipy.ndimage.filters.gaussian_filter(image, sigma)

# # downsample using bilinear interpolation
# def BilinearDownsample(image, factor):
#     # get the current yres and xres
#     yres, xres = image.shape
#     down_yres, down_xres = (int(yres * factor), int(xres * factor))

#     downsampled = np.zeros((down_yres, down_xres), dtype=np.float32)

#     for iy in range(down_yres):
#         for ix in range(down_xres):
#             # get the upsampled y and x values
#             upy = iy / factor
#             upx = ix / factor

#             # use bilinear interpolation
#             alphax = ceil(upx) - upx
#             alphay = ceil(upy) - upy

#             # downsample the image
#             downsampled[iy,ix] = alphax * alphay * image[floor(upy),floor(upx)] + alphax * (1 - alphay) * image[ceil(upy),floor(upx)] + (1 - alphax) * alphay * image[floor(upy),ceil(upx)] + (1 - alphax) * (1 - alphay) * image[ceil(upy),ceil(upx)]

#     return downsampled

# # remove average grayscale from image
# def RemoveDC(grayscale):
#     return grayscale - np.mean(grayscale)

# # extract the features from each valid window
# def CreateKDTree(image, diameter):
#     yres, xres = image.shape

#     # create an empty features array
#     nfeatures = (yres - diameter[0] + 1) * (xres - diameter[1] + 1)
#     ndims = diameter[0] * diameter[1]
#     features = np.zeros((nfeatures, ndims), dtype=np.float32)

#     # generate all features
#     index = 0
#     radius = (diameter[0] / 2, diameter[1] / 2)
#     for iy in range(radius[0], yres - radius[0]):
#         for ix in range(radius[1], xres - radius[1]):
#             features[index,:] = image[iy-radius[0]:iy+radius[0]+1,ix-radius[1]:ix+radius[1]+1].flatten()
#             index += 1

#     return KDTree(features)

# # create a hierarchy of images
# def CreateHierachy(image, parameters):
#     # get useful parameters and variables
#     m = parameters['m']
#     n = parameters['n']
#     scale = parameters['scale']
#     diameter = parameters['diameter']

#     nlayers = m + n + 1

#     # create empty variables for the grayscale images and the kdtrees for features
#     grayscales = [ _ for _ in range(nlayers)]
#     kdtrees = [ _ for _ in range(nlayers)]

#     # create the hierarchy of images
#     grayscales[0] = RGB2Gray(image)
#     for iv in range(-1, -(m + 1), -1):
#         sigma = math.sqrt(pow(scale, -1 * iv))
#         factor = pow(scale, iv)
#         grayscales[iv] = BilinearDownsample(GaussianBlur(grayscales[0], sigma), factor)

#     # remove the DC components from every image
#     grayscales[0] = RemoveDC(grayscales[0])
#     for iv in range(-1, -(m + 1), -1):
#         grayscales[iv] = RemoveDC(grayscales[iv])

#     # create the kdtree for every known layer
#     for iv in range(-m, 1):
#         kdtrees[iv] = CreateKDTree(grayscales[iv], diameter)

#     return grayscales, kdtrees

# # shift the image by x and y pixels
# def SubpixelShift(images, xshift, yshift, parameters):
#     # get useful parameters and variables
#     m = parameters['m']
#     n = parameters['n']
#     nlayers = m + n + 1

#     shifts = [ _ for _ in range(nlayers)]
#     for ie in range(-m, 1):
#         shifts[ie] = scipy.ndimage.interpolation.shift(images[ie], (yshift, xshift))

#     # return the shifted image
#     return shifts

# # get the distance threshold for each image
# def DistanceThreshold(images, shifts, parameters):
#     # get useful parameters and variables
#     m = parameters['m']
#     n = parameters['n']
#     diameter = parameters['diameter']
#     radius = (diameter[0] / 2, diameter[1] / 2)

#     nlayers = m + n + 1
#     gaussian_kernel = Gaussian(diameter, 1.5).flatten()
#     # go through every image
#     distances = [ _ for _ in range(nlayers)]
#     for ie in range(-m , 1):
#         yres, xres = images[ie].shape
#         distances[ie] = np.zeros((yres, xres), dtype=np.float32)

#         # go through each element in this image
#         for iy in range(radius[0], yres - radius[0]):
#             for ix in range(radius[1], xres - radius[1]):
#                 # get the features for the normal and shifted image
#                 image_feature = images[ie][iy-radius[0]:iy+radius[0]+1,ix-radius[1]:ix+radius[1]+1].flatten()
#                 shift_feature = shifts[ie][iy-radius[0]:iy+radius[0]+1,ix-radius[1]:ix+radius[1]+1].flatten()
                
#                 # get the gaussian weighted SSD difference
#                 distances[ie][iy,ix] = np.sum(np.multiply(np.multiply(image_feature - shift_feature, image_feature - shift_feature), gaussian_kernel))

#     return distances

# # convert the index to indices
# @jit(nopython=True)
# def IndexToIndices(index, shape, diameter):
#     # get the correct shape for this value
#     xres = shape[1] - diameter[1] + 1
#     iy = index / xres
#     ix = index % xres
#     return iy + diameter[0] / 2, ix + diameter[1] / 2

# @jit(nopython=True)
# def IndicesToIndex(ix, iy, shape):
#     return iy * shape[1] + ix

# class Constraint:
#     def __init__(self, highy, highx, ys, xs, value, sigma):
#         self.highy = highy
#         self.highx = highx
#         self.ys = ys
#         self.xs = xs
#         self.value = value
#         self.sigma = sigma

# # get the next level 
# def Upsample(grayscales, kdtrees, n, parameters):
#     # get useful parameters and variables
#     scale = parameters['scale']
#     diameter = parameters['diameter']
#     radius = (diameter[0] / 2, diameter[1] / 2)

#     # create an empty high res image
#     yres, xres = grayscales[0].shape
#     magnification = pow(scale, n)
#     high_yres, high_xres = (ceil(yres * magnification), ceil(xres * magnification))
#     grayscales[n] = np.zeros((high_yres, high_xres), dtype=np.float32)


#     # consider all high res images from 0 to level
#     for ii in range(1):
#         # get this image
#         image = grayscales[ii]

#         # get the size of this image
#         yres, xres = image.shape

#         # get the difference between n and this level
#         delta = n - ii

#         # get the low resolution image
#         lowres_image_shape = grayscales[-1 * delta].shape
#         lowres_kdtree = kdtrees[-1 * delta]

#         # get the scale from the lowres image to image
#         magnification = pow(scale, delta)

#         # iterate through all valid locations
#         for iy in range(radius[0], yres - radius[0]):
#             print iy
#             import sys
#             sys.stdout.flush()
#             for ix in range(radius[1], xres - radius[1]):
#                 # get the feature at this location
#                 feature = image[iy-radius[0]:iy+radius[0]+1,ix-radius[0]:ix+radius[1]+1].flatten()

#                 # search for the closest feature in the lowres_image
#                 _, location = lowres_kdtree.query(feature, 1)

#                 lowres_y, lowres_x = IndexToIndices(location, lowres_image_shape, diameter)
#                 high_radius = (floor(diameter[0] / 2 * magnification), floor(diameter[1] / 2 * magnification))
#                 highy, highx = (int(round(lowres_y * magnification)), int(round(lowres_x * magnification)))

#                 # scale iy and ix
#                 scaled_iy = int(round(magnification * iy))
#                 scaled_ix = int(round(magnification * ix))

#                 # get the corresponding image section
#                 grayscales[n][scaled_iy-high_radius[0]:scaled_iy+high_radius[0]+1,scaled_ix-high_radius[1]:scaled_ix+high_radius[1]+1] = image[highy-high_radius[0]:highy+high_radius[0]+1,highx-high_radius[1]:highx+high_radius[1]+1]
#     kdtrees[n] = CreateKDTree(grayscales[n], diameter)

# # calculate the single image super resolution
# def SingleImageSR(root_filename, parameters):
#     # get useful parameters and variables
#     m = parameters['m']
#     n = parameters['n']
#     scale = parameters['scale']

#     # total number of layers
#     nlayers = m + n + 1

#     # read the image into a float32 array between [0, 1]
#     image = dataIO.ReadImage('{}.png'.format(root_filename))
#     yres, xres, depth = image.shape
#     assert (depth == 3)

#     # create the hierarchy of images
#     grayscales, kdtrees = CreateHierachy(image, parameters)

#     # create a shift of the input image
#     shifts = SubpixelShift(grayscales, 0.5, 0.0, parameters)

#     # get the distance for all images
#     distances = DistanceThreshold(grayscales, shifts, parameters)

#     #Upsample(grayscales, kdtrees, 1, parameters)
#     #dataIO.WriteImage('output-1.png', grayscales[1])
#     #Upsample(grayscales, kdtrees, 2, parameters)
#     #dataIO.WriteImage('output-2.png', grayscales[2])
#     Upsample(grayscales, kdtrees, 3, parameters)
#     dataIO.WriteImage('output-3.png', grayscales[3])
#     Upsample(grayscales, kdtrees, 4, parameters)
#     dataIO.WriteImage('output-4.png', grayscales[4])
#     Upsample(grayscales, kdtrees, 5, parameters)
#     dataIO.WriteImage('output-5.png', grayscales[5])
#     Upsample(grayscales, kdtrees, 6, parameters)
#     dataIO.WriteImage('output-6.png', grayscales[6])
    
    
    
    
    
    
#     # run nearest neighbor and bicubic interpolation
#     magnification = pow(scale, n)
#     high_yres, high_xres = (ceil(yres * magnification), ceil(xres * magnification))

#     nearest_neighbor = skimage.transform.resize(image, (high_yres, high_xres), order=0, mode='reflect')
#     bicubic_interpolation = skimage.transform.resize(image, (high_yres, high_xres), order=3, mode='reflect')

#     nearest_filename = '{}-nearest-neighbor.png'.format(root_filename)
#     bicubic_filename = '{}-bicubic-interpolation.png'.format(root_filename)

#     dataIO.WriteImage(nearest_filename, nearest_neighbor)
#     dataIO.WriteImage(bicubic_filename, bicubic_interpolation)


# #################################
# #### VISUALIZATION FUNCTIONS ####
# #################################

# def BoundingBox(output_filename, image, ix, iy, diameter):
#     radius = (diameter[0] / 2, diameter[1] / 2)

#     # get the high and low parameters for the image
#     xlow = (ix - radius[1] - 1)
#     xhigh = (ix + radius[1] + 1)
#     ylow = (iy - radius[0] - 1)
#     yhigh = (iy + radius[0] + 1)

#     image[ylow:yhigh+1,xlow] = 1
#     image[ylow:yhigh+1,xhigh] = 1
#     image[ylow,xlow:xhigh+1] = 1
#     image[yhigh,xlow:xhigh+1] = 1

#     dataIO.WriteImage(output_filename, image)

# def FeatureMatching(root_filename, grayscales, kdtrees, parameters):
#     # get useful parameters and variables
#     diameter = parameters['diameter']
#     scale = parameters['scale']

#     yres, xres = grayscales[0].shape  
#     radius = (diameter[0] / 2, diameter[1] / 2)

#     ix = 80
#     iy = 80

#     # get this feature
#     feature = grayscales[0][iy-radius[0]:iy+radius[0]+1,ix-radius[1]:ix+radius[1]+1].flatten()

#     # find the closest feature in the next lowest level
#     value, location = kdtrees[-1].query(feature, 1)

#     # get the coordinate in this lower level
#     lowy, lowx = IndexToIndices(location, grayscales[-1].shape, diameter)

#     BoundingBox('{}-bounding.png'.format(root_filename), grayscales[0], ix, iy, diameter)
#     BoundingBox('{}-bounding-low.png'.format(root_filename), grayscales[-1], lowx, lowy, diameter)

#     # get the difference between the scales
#     increase = pow(scale, 1)
#     high_diameter = (floor(diameter[0] * increase), floor(diameter[1] * increase))
#     highy, highx = (int(round(lowy * increase)), int(round(lowx * increase)))
#     BoundingBox('{}-bounding-high.png'.format(root_filename), grayscales[0], highx, highy, high_diameter)

