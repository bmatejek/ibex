import math
import sys

import numpy as np

# define conveniet ceil and floor functions
def ceil(value): return int(math.ceil(value))
def floor(value): return int(math.floor(value))

# conversion between linear and quadric spaces
def IndicesToIndex(iy, ix, xres): return iy * xres + ix
def IndexToIndices(index, xres): return (index / xres, index % xres)

# remove average intensity from image
def RemoveDC(intensity):
    return intensity - np.mean(intensity)

# create a gaussian kernel
def SSDGaussian(diameter):
    sys.stderr.write('Warning: using diameter of ({}, {}) for SSD\n'.format(diameter[0], diameter[1]))
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
def ExtractFeature(intensities, iy, ix, diameter, weighted=True):
    # get convenient variables
    radius = diameter / 2
    yres, xres = intensities.shape

    # see if reflection is needed
    if iy > radius - 1 and ix > radius - 1 and iy < yres - radius and ix < xres - radius:
        if weighted: return np.multiply(ssd_kernel, intensities[iy-radius:iy+radius+1,ix-radius:ix+radius+1].flatten())
        else: return intensities[iy-radius:iy+radius+1,ix-radius:ix+radius+1].flatten()
    else: 
        return sys.maxint * np.ones((diameter, diameter), dtype=np.float32).flatten()
    