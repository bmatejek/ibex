import os
import h5py
import numpy as np
from ibex.data_structures import meta_data, skeleton
from ibex.utilities.constants import *
from PIL import Image
import imageio
import tifffile


def GetWorldBBox(prefix):
    # return the bounding box for this segment
    return meta_data.MetaData(prefix).WorldBBox()



def ReadMetaData(prefix):
    # return the meta data for this prefix
    return meta_data.MetaData(prefix)



def Resolution(prefix):
    # return the resolution for this prefix
    return meta_data.MetaData(prefix).Resolution()



def ReadH5File(filename, dataset=None):
    # read the h5py file
    with h5py.File(filename, 'r') as hf:
        # read the first dataset if none given
        if dataset == None: data = np.array(hf[hf.keys()[0]])
        else: data = np.array(hf[dataset])

    # return the data
    return data



def IsIsotropic(prefix):
    resolution = Resolution(prefix)
    return (resolution[IB_Z] == resolution[IB_Y]) and (resolution[IB_Z] == resolution[IB_X])


def WriteH5File(data, filename, dataset):
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset(dataset, data=data)


def ReadAffinityData(prefix):
    filename, dataset = meta_data.MetaData(prefix).AffinityFilename()

    return ReadH5File(filename, dataset).astype(np.float32)


def ReadSegmentationData(prefix):
    filename, dataset = meta_data.MetaData(prefix).SegmentationFilename()

    return ReadH5File(filename, dataset).astype(np.int64)



def ReadGoldData(prefix):
    filename, dataset = meta_data.MetaData(prefix).GoldFilename()

    return ReadH5File(filename, dataset).astype(np.int64)



def ReadImageData(prefix):
    filename, dataset = meta_data.MetaData(prefix).ImageFilename()

    return ReadH5File(filename, dataset)



def ReadSkeletons(prefix, skeleton_algorithm='thinning', downsample_resolution=(100, 100, 100), benchmark=False):
    skeletons = Skeletons(prefix, skeleton_algorithm, downsample_resolution, benchmark)

    return skeletons



def ReadImage(filename):
    return np.array(Image.open(filename))



def WriteImage(filename, image):
   imageio.imwrite(filename, image)



def H52Tiff(stack, output_prefix):
    zres, _, _ = stack.shape

    for iz in range(zres):
        image = stack[iz,:,:]
        tifffile.imsave('{}-{:05d}.tif'.format(output_prefix, iz), image)



def H52PNG(stack, output_prefix):
    zres, _, _ = stack.shape

    for iz in range(zres):
        image = stack[iz,:,:]
        im = Image.fromarray(image)
        im.save('{}-{:05d}.png'.format(output_prefix, iz))



def PNG2H5(directory, filename, dataset, dtype=np.int32):
    # get all of the png files
    png_files = sorted(os.listdir(directory))

    # what is the size of the output file
    zres = len(png_files)
    for iz, png_filename in enumerate(png_files):
        im = np.array(Image.open('{}/{}'.format(directory, png_filename)))

        # create the output if this is the first slice
        if not iz:
            if len(im.shape) == 2: yres, xres = im.shape
            else: yres, xres, _ = im.shape
            
            h5output = np.zeros((zres, yres, xres), dtype=dtype)

        # add this element
        if len(im.shape) == 3 and dtype == np.int32: h5output[iz,:,:] = 65536 * im[:,:,0] + 256 * im[:,:,1] + im[:,:,2]
        elif len(im.shape) == 3 and dtype == np.uint8: h5output[iz,:,:] = ((im[:,:,0].astype(np.uint16) + im[:,:,1].astype(np.uint16) + im[:,:,2].astype(np.uint16)) / 3).astype(np.uint8)
        else: h5output[iz,:,:] = im[:,:]

    WriteH5File(h5output, filename, dataset)
