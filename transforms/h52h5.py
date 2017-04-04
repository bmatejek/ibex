# general functions for transforming h5 files
from numba import jit


# downsample the data by (z, y, x) ratio
@jit(nopython=True)
def DownsampleData(data, ratio=(1, 2, 2)):
    # get the size of the current dataset
    (zres, yres, xres) = data.shape

    # create an empty array for sownsampling
    (down_zres, down_yres, down_xres) = (zres / ratio[0], yres / ratio[1], xres / ratio[2])
    downsampled_data = np.zeros((zres / ratio[0], yres / ratio[1], xres / ratio[2]), dtype=data.dtype)

    # fill in the entries of the array
    for iz in range(down_zres):
        for iy in range(down_yres):
            for ix in range(down_xres):
                downsampled_data[iz,iy,ix] = data[iz * ratio[0] + 1, iy * ratio[1] + 1, ix * ratio[2] + 1]

    return downsampled_data



# split the data to create training and validation data
@jit(nopython=True)
def CreateTrainValidation(data, threshold=0.5):
    # get the separation idnex
    separation = int(threshold * data.shape[0])

    # split the data into two components
    training_data = data[0:separation,:,:]
    validation_data = data[separation:,:,:]

    # return the training and validation data
    return training_data, validation_data