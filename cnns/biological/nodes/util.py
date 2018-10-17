import random
import numpy as np

from numba import jit



@jit(nopython=True)
def ScaleFeature(segment, width):
    # get the size of the extracted segment
    zres, yres, xres = segment.shape
    nchannels = width[0]

    example = np.zeros((1, nchannels, width[IB_Z + 1], width[IB_Y + 1], width[IB_X + 1]), dtype=np.float32)

    # iterate over the example coordinates
    for iz in range(width[IB_Z + 1]):
        for iy in range(width[IB_Y + 1]):
            for ix in range(width[IB_X + 1]):
                # get the global coordiantes from segment
                iw = int(float(zres) / float(width[IB_Z + 1]) * iz)
                iv = int(float(yres) / float(width[IB_Y + 1]) * iy)
                iu = int(float(xres) / float(width[IB_X + 1]) * ix)

                if nchannels == 1 and segment[iw,iv,iu]:
                    example[0,0,iz,iy,ix] = 1
                else:
                    # add second channel
                    if segment[iw,iv,iu] == 1:
                        example[0,0,iz,iy,ix] = 1
                    elif segment[iw,iv,iu] == 2:
                        example[0,1,iz,iy,ix] = 1
                    # add third channel 
                    if nchannels == 3 and (segment[iw,iv,iu] == 1 or segment[iw,iv,iu] == 2):
                        example[0,2,iz,iy,ix] = 1

    example = example - 0.5

    return example



def AugmentFeature(segment, width):
    example = ScaleFeature(segment, width)

    if random.random() > 0.5: example = np.flip(example, IB_Z + 2)

    angle = random.uniform(0, 360)
    example = scipy.ndimage.interpolation.rotate(example, angle, axes=(IB_X + 2, IB_Y + 2), reshape=False, order=0, mode='constant', cval=-0.5)
    
    return example