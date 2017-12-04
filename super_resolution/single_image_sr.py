import math
import numpy as np

from numba import jit
from scipy.spatial import KDTree
import scipy.ndimage
import scipy.signal

from ibex.utilities import dataIO



# small utility functions
def ceil(value): return int(math.ceil(value))
def floor(value): return int(math.floor(value))



# convert the image to grayscale
def RGB2Gray(image):
    yres, xres, depth = image.shape
    assert (depth == 3)
    # G = 0.2126 R + 0.7152 G + 0.0722 B
    grayscale = 0.2126 * image[:,:,2] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,0]
    return grayscale



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


@jit(nopython=True)
def IndexToIndices(index, shape, diameter):
    # get the correct shape for this value
    xres = shape[1] - diameter[1] + 1

    iy = index / xres
    ix = index % xres

    return iy + diameter[0] / 2, ix + diameter[1] / 2


class Constraint:
    def __init__(self, midy, midx, lowy, lowx, midlevel, highlevel, value):
        self.midy = midy
        self.midx = midx
        self.lowy = lowy
        self.lowx = lowx
        self.midlevel = midlevel
        self.highlevel = highlevel
        self.value = value



def Upsample(grayscales, kdtrees, level, parameters):
    # get useful parameters
    m = parameters['m']
    scale = parameters['scale']
    diameter = parameters['diameter']
    radius = (diameter[0] / 2, diameter[1] / 2)

    # get the original input size
    yres, xres = grayscales[level - 1].shape
    scale_factor = pow(scale, level)
    high_yres, high_xres = (floor(yres * scale_factor), floor(xres * scale_factor))

    # create an empty grayscale level
    grayscales[level] = np.zeros((high_yres, high_xres), dtype=np.float32)

    constraints = []

    # iterate through all of the high res images
    for iv in range(level):
        input_yres, input_xres = grayscales[iv].shape

        # what is the difference between this level and the target level
        diff = iv - level

        # iterate over all pixels in this image
        for midy in range(radius[0], input_yres - radius[0]):
            for midx in range(radius[1], input_xres - radius[1]):
                # get the feature at this level
                feature = grayscales[iv][midy-radius[0]:midy+radius[0]+1,midx-radius[1]:midx+radius[1]+1].flatten()

                # look in the corresponding low resolution image for nearby features
                value, location = kdtrees[diff].query(feature, 1)
                lowy, lowx = IndexToIndices(location, grayscales[diff].shape, diameter)

                constraints.append(Constraint(midy, midx, lowy, lowx, iv, level, grayscales[iv][midy,midx]))

    # solve the system of constraints
    nconstraints = len(constraints)
    H = np.zeros((high_yres * high_xres, nconstraints), dtype=np.float32)
    b = np.zeros(nconstraints, dtype=np.float32)

    # populate H for every constraint
    for ie, constraint in enumerate(constraints):


        # set the value
        b[ie] = constraint.value



    import sys
    sys.exit()


    #kdtrees[level] = CreateKDTree(grayscales[level], diameter)


# calculate the single image super resolution
def SingleImageSR(root_filename, parameters):
    # get useful parameters and variables
    m = parameters['m']
    n = parameters['n']
    diameter = parameters['diameter']
    scale = parameters['scale']

    # total number of layers
    nlayers = m + n + 1



    # read the image into a float32 array between [0, 1]
    image = dataIO.ReadImage('{}.png'.format(root_filename))
    image = image[0:40,0:40,:]

    yres, xres, depth = image.shape
    assert (depth == 3)

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



    # create all of the images from 1 to n
    for iv in range(1, n + 1):
        Upsample(grayscales, kdtrees, iv, parameters)




    for iv in range(-1, -(m + 1), -1):
        grayscale_filename = '{}{}.png'.format(root_filename, iv)
        dataIO.WriteImage(grayscale_filename, grayscales[iv])
    import sys
    sys.exit()




















    # output all of the images
    for iv in range(0, n + 1):
        grayscale_filename = '{}+{}.png'.format(root_filename, iv)
        dataIO.WriteImage(grayscale_filename, grayscales[iv])






















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
