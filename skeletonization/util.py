import numpy as np
from numba import jit



@jit(nopython=True)
def PreprocessSegment(segmentation, label):
    zres, yres, xres = segmentation.shape
    zmin, ymin, xmin = (zres, yres, xres)
    zmax, ymax, xmax = (0, 0, 0)

    # find the minimum and maximum span for this element
    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if not segmentation[iz,iy,ix] == label: continue

                if iz < zmin: zmin = iz
                if iy < ymin: ymin = iy
                if ix < xmin: xmin = ix
                if iz > zmax: zmax = iz
                if iy > ymax: ymax = iy
                if ix > xmax: xmax = ix

    # generate a binary image for this label
    preprocessed_segmentation = np.copy(segmentation[zmin:zmax+1,ymin:ymax+1,xmin:xmax+1])
    zres, yres, xres = preprocessed_segmentation.shape

    for iz in range(zres):
        for iy in range(yres):
            for ix in range(xres):
                if preprocessed_segmentation[iz,iy,ix] == label: preprocessed_segmentation[iz,iy,ix] = 1
                else: preprocessed_segmentation[iz,iy,ix] = 0

    return zmin, ymin, xmin, preprocessed_segmentation

