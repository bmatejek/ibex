import math
import numpy as np

from numba import jit
from scipy.spatial import KDTree
import scipy.ndimage
import scipy.signal

from ibex.utilities import dataIO



# small utility functions
@jit(nopython=True)
def ceil(value): return int(math.ceil(value))
@jit(nopython=True)
def floor(value): return int(math.floor(value))

def Gaussian(diameter, sigma):
    radius = (diameter[0] / 2, diameter[1] / 2)
    y_range = np.arange(-radius[0], radius[0] + 1)
    x_range = np.arange(-radius[1], radius[1] + 1)
    
    xx, yy = np.meshgrid(x_range, y_range)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)) 
    kernel = kernel / np.sum(kernel)

    return kernel

# convert the image to grayscale
def RGB2Gray(image):
    yres, xres, depth = image.shape
    assert (depth == 3)
    # G = 0.2126 R + 0.7152 G + 0.0722 B
    grayscale = 0.2126 * image[:,:,2] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,0]
    return grayscale

# apply a gaussian blue
def GaussianBlur(image, sigma):
    return scipy.ndimage.filters.gaussian_filter(image, sigma)

# downsample using bilinear interpolation
def BilinearDownsample(image, factor):
    # get the current yres and xres
    yres, xres = image.shape
    down_yres, down_xres = (int(yres * factor), int(xres * factor))

    downsampled = np.zeros((down_yres, down_xres), dtype=np.float32)

    for iy in range(down_yres):
        for ix in range(down_xres):
            # get the upsampled y and x values
            upy = iy / factor
            upx = ix / factor

            # use bilinear interpolation
            alphax = ceil(upx) - upx
            alphay = ceil(upy) - upy

            # downsample the image
            downsampled[iy,ix] = alphax * alphay * image[floor(upy),floor(upx)] + alphax * (1 - alphay) * image[ceil(upy),floor(upx)] + (1 - alphax) * alphay * image[floor(upy),ceil(upx)] + (1 - alphax) * (1 - alphay) * image[ceil(upy),ceil(upx)]

    return downsampled

# remove average grayscale from image
def RemoveDC(grayscale):
    return grayscale - np.mean(grayscale)

# extract the features from each valid window
def CreateKDTree(image, diameter):
    yres, xres = image.shape

    # create an empty features array
    nfeatures = (yres - diameter[0] + 1) * (xres - diameter[1] + 1)
    ndims = diameter[0] * diameter[1]
    features = np.zeros((nfeatures, ndims), dtype=np.float32)

    # generate all features
    index = 0
    radius = (diameter[0] / 2, diameter[1] / 2)
    for iy in range(radius[0], yres - radius[0]):
        for ix in range(radius[1], xres - radius[1]):
            features[index,:] = image[iy-radius[0]:iy+radius[0]+1,ix-radius[1]:ix+radius[1]+1].flatten()
            index += 1

    return KDTree(features)

# create a hierarchy of images
def CreateHierachy(image, parameters):
    # get useful parameters and variables
    m = parameters['m']
    n = parameters['n']
    scale = parameters['scale']
    diameter = parameters['diameter']

    nlayers = m + n + 1

    # create empty variables for the grayscale images and the kdtrees for features
    grayscales = [ _ for _ in range(nlayers)]
    kdtrees = [ _ for _ in range(nlayers)]

    # create the hierarchy of images
    grayscales[0] = RGB2Gray(image)
    for iv in range(-1, -(m + 1), -1):
        sigma = math.sqrt(pow(scale, -1 * iv))
        factor = pow(scale, iv)
        grayscales[iv] = BilinearDownsample(GaussianBlur(grayscales[0], sigma), factor)

    # remove the DC components from every image
    grayscales[0] = RemoveDC(grayscales[0])
    for iv in range(-1, -(m + 1), -1):
        grayscales[iv] = RemoveDC(grayscales[iv])

    # create the kdtree for every known layer
    for iv in range(-m, 1):
        kdtrees[iv] = CreateKDTree(grayscales[iv], diameter)

    return grayscales, kdtrees

# shift the image by x and y pixels
def SubpixelShift(images, xshift, yshift, parameters):
    # get useful parameters and variables
    m = parameters['m']
    n = parameters['n']
    nlayers = m + n + 1

    shifts = [ _ for _ in range(nlayers)]
    for ie in range(-m, 1):
        shifts[ie] = scipy.ndimage.interpolation.shift(images[ie], (yshift, xshift))

    # return the shifted image
    return shifts

# get the distance threshold for each image
def DistanceThreshold(images, shifts, parameters):
    # get useful parameters and variables
    m = parameters['m']
    n = parameters['n']
    diameter = parameters['diameter']
    radius = (diameter[0] / 2, diameter[1] / 2)

    nlayers = m + n + 1
    gaussian_kernel = Gaussian(diameter, 1.5).flatten()
    # go through every image
    distances = [ _ for _ in range(nlayers)]
    for ie in range(-m , 1):
        yres, xres = images[ie].shape
        distances[ie] = np.zeros((yres, xres), dtype=np.float32)

        # go through each element in this image
        for iy in range(radius[0], yres - radius[0]):
            for ix in range(radius[1], xres - radius[1]):
                # get the features for the normal and shifted image
                image_feature = images[ie][iy-radius[0]:iy+radius[0]+1,ix-radius[1]:ix+radius[1]+1].flatten()
                shift_feature = shifts[ie][iy-radius[0]:iy+radius[0]+1,ix-radius[1]:ix+radius[1]+1].flatten()
                
                # get the gaussian weighted SSD difference
                distances[ie][iy,ix] = np.sum(np.multiply(np.multiply(image_feature - shift_feature, image_feature - shift_feature), gaussian_kernel))

    return distances

# convert the index to indices
@jit(nopython=True)
def IndexToIndices(index, shape, diameter):
    # get the correct shape for this value
    xres = shape[1] - diameter[1] + 1
    iy = index / xres
    ix = index % xres
    return iy + diameter[0] / 2, ix + diameter[1] / 2

@jit(nopython=True)
def IndicesToIndex(ix, iy, shape):
    return iy * shape[1] + ix

class Constraint:
    def __init__(self, highy, highx, ys, xs, value, sigma):
        self.highy = highy
        self.highx = highx
        self.ys = ys
        self.xs = xs
        self.value = value
        self.sigma = sigma

# get the next level 
def Upsample(grayscales, kdtrees, n, parameters):
    # get useful parameters and variables
    scale = parameters['scale']
    diameter = parameters['diameter']
    radius = (diameter[0] / 2, diameter[1] / 2)

    # create an empty high res image
    yres, xres = grayscales[0].shape
    magnification = pow(scale, n)
    high_yres, high_xres = (ceil(yres * magnification), ceil(xres * magnification))
    grayscales[n] = np.zeros((high_yres, high_xres), dtype=np.float32)


    # consider all high res images from 0 to level
    for ii in range(n):
        # get this image
        image = grayscales[ii]

        # get the size of this image
        yres, xres = image.shape

        # get the difference between n and this level
        delta = n - ii

        # get the low resolution image
        lowres_image_shape = grayscales[-1 * delta].shape
        lowres_kdtree = kdtrees[-1 * delta]

        # get the scale from the lowres image to image
        magnification = pow(scale, delta)

        # iterate through all valid locations
        for iy in range(radius[0], yres - radius[0]):
            print iy
            for ix in range(radius[1], xres - radius[1]):
                # get the feature at this location
                feature = image[iy-radius[0]:iy+radius[0]+1,ix-radius[0]:ix+radius[1]+1].flatten()

                # search for the closest feature in the lowres_image
                _, location = lowres_kdtree.query(feature, 1)

                lowres_y, lowres_x = IndexToIndices(location, lowres_image_shape, diameter)
                high_radius = (floor(diameter[0] * magnification / 2), floor(diameter[1] * magnification / 2))
                highy, highx = (int(round(lowres_y * magnification)), int(round(lowres_x * magnification)))

                # scale iy and ix
                scaled_iy = int(round(magnification * iy))
                scaled_ix = int(round(magnification * ix))

                # get the corresponding image section
                grayscales[n][scaled_iy-high_radius[0]:scaled_iy+high_radius[0]+1,scaled_ix-high_radius[1]:scaled_ix+high_radius[1]+1] = image[highy-high_radius[0]:highy+high_radius[0]+1,highx-high_radius[1]:highx+high_radius[1]+1]



















# calculate the single image super resolution
def SingleImageSR(root_filename, parameters):
    # get useful parameters and variables
    m = parameters['m']
    n = parameters['n']

    # total number of layers
    nlayers = m + n + 1

    # read the image into a float32 array between [0, 1]
    image = dataIO.ReadImage('{}.png'.format(root_filename))
    yres, xres, depth = image.shape
    assert (depth == 3)

    # create the hierarchy of images
    grayscales, kdtrees = CreateHierachy(image, parameters)

    # create a shift of the input image
    shifts = SubpixelShift(grayscales, 0.5, 0.0, parameters)

    # get the distance for all images
    distances = DistanceThreshold(grayscales, shifts, parameters)

    #FeatureMatching(root_filename, grayscales, kdtrees, parameters)


    Upsample(grayscales, kdtrees, 1, parameters)

    dataIO.WriteImage('output.png', grayscales[1])


    # # create all of the images from 1 to n
    # for iv in range(1, n + 1):
    #     Upsample(grayscales, kdtrees, iv, parameters)


    #dataIO.WriteImage('output.jpg', grayscales[1])

    # for iv in range(-1, -(m + 1), -1):
    #     grayscale_filename = '{}{}.png'.format(root_filename, iv)
    #     dataIO.WriteImage(grayscale_filename, grayscales[iv])
    #     shift_filename = '{}{}shift.png'.format(root_filename, iv)
    #     dataIO.WriteImage(shift_filename, shifts[iv])
    #     distance_filename = '{}{}distance.png'.format(root_filename, iv)
    #     dataIO.WriteImage(distance_filename, distances[iv])
    # import sys
    # sys.exit()

#################################
#### VISUALIZATION FUNCTIONS ####
#################################

def BoundingBox(output_filename, image, ix, iy, diameter):
    radius = (diameter[0] / 2, diameter[1] / 2)

    # get the high and low parameters for the image
    xlow = (ix - radius[1] - 1)
    xhigh = (ix + radius[1] + 1)
    ylow = (iy - radius[0] - 1)
    yhigh = (iy + radius[0] + 1)

    image[ylow:yhigh+1,xlow] = 1
    image[ylow:yhigh+1,xhigh] = 1
    image[ylow,xlow:xhigh+1] = 1
    image[yhigh,xlow:xhigh+1] = 1

    dataIO.WriteImage(output_filename, image)

def FeatureMatching(root_filename, grayscales, kdtrees, parameters):
    # get useful parameters and variables
    diameter = parameters['diameter']
    scale = parameters['scale']

    yres, xres = grayscales[0].shape  
    radius = (diameter[0] / 2, diameter[1] / 2)

    ix = 80
    iy = 80

    # get this feature
    feature = grayscales[0][iy-radius[0]:iy+radius[0]+1,ix-radius[1]:ix+radius[1]+1].flatten()

    # find the closest feature in the next lowest level
    value, location = kdtrees[-1].query(feature, 1)

    # get the coordinate in this lower level
    lowy, lowx = IndexToIndices(location, grayscales[-1].shape, diameter)

    BoundingBox('{}-bounding.png'.format(root_filename), grayscales[0], ix, iy, diameter)
    BoundingBox('{}-bounding-low.png'.format(root_filename), grayscales[-1], lowx, lowy, diameter)

    # get the difference between the scales
    increase = pow(scale, 1)
    high_diameter = (floor(diameter[0] * increase), floor(diameter[1] * increase))
    highy, highx = (int(round(lowy * increase)), int(round(lowx * increase)))
    BoundingBox('{}-bounding-high.png'.format(root_filename), grayscales[0], highx, highy, high_diameter)





    # # output all of the images
    # for iv in range(0, n + 1):
    #     grayscale_filename = '{}+{}.png'.format(root_filename, iv)
    #     dataIO.WriteImage(grayscale_filename, grayscales[iv])






















def NearestNeighborInterpolation(input_filename, scale):
    image = dataIO.ReadImage(input_filename)
    yres, xres, depth = image.shape
    assert (depth == 3)

    # allocate array for output
    oyres, oxres = (yres * scale, xres * scale)
    output_image = np.zeros((oyres, oxres, depth), dtype=np.float32)

    # go through every voxel
    for iy in range(oyres):
        for ix in range(oxres):
            for ie in range(depth):
                # get the closest value in the downsample image
                iv = int(iy / float(scale))
                iu = int(ix / float(scale))

                # assign this output value
                output_image[iy,ix,ie] = image[iv,iu,ie]

    output_filename = '{}-nearest-neighbor.png'.format(input_filename.split('.')[0])
    dataIO.WriteImage(output_filename, output_image)



def BicubicInterpolation(input_filename, scale):
    image = dataIO.ReadImage(input_filename)
    if len(image.shape) == 3:
        yres, xres, depth = image.shape
        assert (depth == 3)
        grayscale = RGB2Gray(image)
    else:
        yres, xres = image.shape
        depth = 1
        grayscale = image


    # allocate array for output
    oyres, oxres = (yres * scale, xres * scale)
    output_image = np.zeros((oyres, oxres), dtype=np.float32)

    # get derivative of the input image 
    dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    dy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # convolve in the x and y dimensions
    f = grayscale
    fx = scipy.signal.convolve2d(grayscale, dx, mode='same')
    fy = scipy.signal.convolve2d(grayscale, dy, mode='same')
    fxy = scipy.signal.convolve2d(fy, dx, mode='same')
    

    aiis = np.zeros((yres, xres, 4, 4), dtype=np.float32)

    # create a00, a01 ... a33 for every block
    for iy in range(yres - 1):
        for ix in range(xres - 1):
            a1 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [-3, 3, -2, -1], [2, -2, 1, 1]])
            A = np.array([[f[iy,ix], f[iy+1,ix], fy[iy,ix], fy[iy+1,ix]], [f[iy,ix+1], f[iy+1,ix+1], fy[iy,ix+1], fy[iy+1,ix+1]], [fx[iy,ix], fx[iy,ix+1], fxy[iy,ix], fxy[iy+1,ix]], [fx[iy,ix+1], fx[iy+1,ix+1], fxy[iy,ix+1], fxy[iy+1,ix+1]]])
            a2 = np.array([[1, 0, -3, -2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])

            aiis[iy,ix,:,:] = np.matmul(np.matmul(a1, A), a2)

    # get bicubic results
    for iy in range(oyres):
        for ix in range(oxres):
            # get the start cube to consider
            iv = iy / float(scale)
            iu = ix / float(scale)

            x = iu - floor(iu)
            y = iv - floor(iv)

            output_image[iy,ix] = np.matmul(np.matmul(np.array([1, y, pow(y,2), pow(y,3)]), aiis[floor(iv),floor(iu),:,:]), np.array([[1], [x], [pow(x, 2)], [pow(x, 3)]]))

    output_filename = '{}-bicubic-interpolation.png'.format(input_filename.split('.')[0])
    dataIO.WriteImage(output_filename, output_image)




# # get useful parameters
#     m = parameters['m']
#     scale = parameters['scale']
#     diameter = parameters['diameter']
#     radius = (diameter[0] / 2, diameter[1] / 2)

#     # get the original input size
#     yres, xres = grayscales[level - 1].shape
#     scale_factor = pow(scale, level)
#     high_yres, high_xres = (floor(yres * scale_factor), floor(xres * scale_factor))

#     # create an empty grayscale level
#     grayscales[level] = np.zeros((high_yres, high_xres), dtype=np.float32)

#     constraints = []

#     # iterate through all of the high res images
#     for iv in range(level):
#         input_yres, input_xres = grayscales[iv].shape
#         l = iv - level

#         # iterate over all pixels in this image
#         for iy in range(radius[0], input_yres - radius[0]):
#             print iy
#             import sys
#             sys.stdout.flush()
#             for ix in range(radius[1], input_xres - radius[1]):
#                 # get the feature at this level
#                 feature = grayscales[iv][iy-radius[0]:iy+radius[0]+1,ix-radius[1]:ix+radius[1]+1].flatten()

#                 # look in the corresponding low resolution image for nearby features
#                 value, location = kdtrees[l].query(feature, 1)
#                 lowy, lowx = IndexToIndices(location, grayscales[l].shape, diameter)

#                 # get the location at this level
#                 thisy = int(round(lowy * pow(scale, -1 * l)))
#                 thisx = int(round(lowx * pow(scale, -1 * l)))
#                 outputx = floor(diameter[0] * pow(scale, -1 * l) / 2)
#                 outputy = floor(diameter[1] * pow(scale, -1 * l) / 2)

#                 highy = iy * pow(scale, -1 * l)
#                 highx = ix * pow(scale, -1 * l)
#                 sigma = math.sqrt(pow(scale, -1 * l))

#                 xs = np.arange(thisx-outputx, thisx+outputx+1)
#                 ys = np.arange(thisy-outputy, thisy+outputy+1)

#                 constraints.append(Constraint(highy, highx, ys, xs, grayscales[iv][iy,ix], sigma))

#     # solve the system of constraints
#     nconstraints = len(constraints)
#     H = np.zeros((nconstraints, high_yres * high_xres), dtype=np.float32)
#     b = np.zeros(nconstraints, dtype=np.float32)

#     # populate H for every constraint
#     for ie, constraint in enumerate(constraints):
#         ys = constraint.ys
#         xs = constraint.xs
#         sigma = constraint.sigma

#         # get this gaussian kernel
#         gaussian_kernel = Gaussian((xs.size, ys.size), sigma)

#         for iv, iy in enumerate(ys):
#             for iu, ix in enumerate(xs):
#                 gaussian_value = gaussian_kernel[iv,iu]

#                 index = IndicesToIndex(ix, iy, (high_yres, high_xres))
#                 H[ie, index] = gaussian_value 

#         # set the value
#         b[ie] = constraint.value


#     print H.shape
#     highres, _ = np.linalg.lstsq(H, b)
#     print highres

#     for iy in range(yres):
#         for ix in range(xres):
#             index = IndicesToIndex(ix, iy, (high_yres, high_xres))
#             grayscales[level][iy,ix] = highres[index]