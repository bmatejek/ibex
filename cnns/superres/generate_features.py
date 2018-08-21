import numpy as np

from ibex.utilities import dataIO
from ibex.utilities.constants import *


def GenerateFeatures(prefix):
    # find the level of anisotropy
    resolution = dataIO.Resolution(prefix)
    zy = resolution[IB_Z] / resolution[IB_Y]
    zx = resolution[IB_Z] / resolution[IB_X]

    # assert isotropy in xy
    assert (zy == zx)

    # read the image 
    image = dataIO.ReadImageData(prefix)
    zres, yres, xres = image.shape

    zx_slice = image[:,0,:]
    zy_slice = image[:,:,0]

    dataIO.WriteImage('{}-zx.png'.format(prefix), zx_slice)
    dataIO.WriteImage('{}-zy.png'.format(prefix), zy_slice)

    
    
    return
    
    for iv in range(zy):
        xcut = image[:,:,iv::zx]
        ycut = image[:,:,iv::zy]

        xfilename = 'super_resolution/{}-x-{}.h5'.format(prefix, iv + 1)
        yfilename = 'super_resolution/{}-y-{}.h5'.format(prefix, iv + 1)
        
        dataIO.WriteH5File(xcut, xfilename, 'main')
        dataIO.WriteH5File(ycut, yfilename, 'main')

        for iz in range(zres):
            xslice = xcut[iz,:,:]
            yslice = ycut[iz,:,:]
            zslice = image[iz,:,:]
            
            xfilename = 'super_resolution/images/{}-x-{}-{:04d}.png'.format(prefix, iv + 1, iz)
            yfilename = 'super_resolution/images/{}-y-{}-{:04d}.png'.format(prefix, iv + 1, iz)
            zfilename = 'super_resolution/images/{}-z-{:04d}.png'.format(prefix, iz)
            
            dataIO.WriteImage(xfilename, xslice)
            dataIO.WriteImage(yfilename, yslice)
            dataIO.WriteImage(zfilename, zslice)
            
        break
