import numpy as np
import imageio

from ibex.utilities.constants import *


def ColorizeSlice(image):
    yres, xres = image.shape

    colored_image = np.zeros((yres, xres, 3), dtype=np.uint8)

    colored_image[:,:,0] = (107 * image % 700) % 255
    colored_image[:,:,1] = (509 * image % 900) % 255
    colored_image[:,:,2] = (200 * image % 777) % 255

    return colored_image


def ColorizeStack(stack):
    zres, yres, xres = stack.shape

    output_stack = np.zeros((zres, yres, xres, 3), dtype=np.uint8)

    for iz in range(zres):
        output_stack[iz,:,:,:] = ColorizeSlice(stack[iz,:,:])
        
    return output_stack


def MakeImage3D(image):
    yres, xres = image.shape

    output_image = np.zeros((yres, xres, 3), dtype=image.dtype)

    output_image[:,:,0] = image
    output_image[:,:,1] = image
    output_image[:,:,2] = image

    return output_image


def MakeStack3D(stack):
    zres, yres, xres = stack.shape

    output_stack = np.zeros((zres, yres, xres, 3), dtype=stack.dtype)
    output_stack[:,:,:,0] = stack
    output_stack[:,:,:,1] = stack
    output_stack[:,:,:,2] = stack

    return output_stack


def H52Gif(filename, stack, duration=0.5, axis=IB_Z):
    images = []
    
    if len(stack.shape) != 4: stack = MakeStack3D(stack)
    zres, yres, xres, _ = stack.shape

    if axis == IB_Z:
        for iz in range(zres):
            images.append(stack[iz,:,:,:])
    elif axis == IB_Y:
        for iy in range(yres):
            images.append(stack[:,iy,:,:])
    elif axis == IB_X:
        for ix in range(xres):
            images.append(stack[:,:,ix,:])
    imageio.mimsave(filename, images, duration=duration)


def Images2Gif(filename, images, duration=0.5):
    imageio.mimsave(filename, images, duration=duration)


def Overlay(segmentation, image, alpha):
    assert (segmentation.shape == image.shape)
    if len(segmentation.shape) == 2:
        segmentation = ColorizeSlice(segmentation)
        image = MakeImage3D(image)
    elif len(segmentation.shape) == 3:
        segmentation = ColorizeStack(segmentation)
        image = MakeStack3D(image)
    else:
        sys.stderr.write('Unrecognized number of dimensions: %d\n'.format(len(segmentation.shape)))
        sys.exit()

    return np.uint8(alpha * segmentation + (1 - alpha) * image)

