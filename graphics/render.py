import numpy as np

from skimage import measure
from stl import mesh

from ibex.utilities.constants import *
from ibex.utilities import dataIO


from numba import jit


@jit(nopython=True)
def Downsample(segmentation, factor):
    zres, yres, xres = segmentation.shape
    newz, newy, newx = (zres / factor[IB_Z], yres / factor[IB_Y], xres / factor[IB_X])

    downsampled_segmentation = np.zeros((newz, newy, newx), dtype=np.int32)

    for iz in range(newz):
        for iy in range(newy):
            for ix in range(newx):
                iw = iz * factor[IB_Z]
                iv = iy * factor[IB_Y]
                iu = ix * factor[IB_X]

                downsampled_segmentation[iz,iy,ix] = segmentation[iw,iv,iu]

    return downsampled_segmentation



# run the marching cube algorithm for these labels
def MarchingCubes(prefix, labels):
    downsample_rate = (1, 2, 2)
    segmentation = Downsample(dataIO.ReadSegmentationData(prefix), downsample_rate)
    resolution = dataIO.Resolution(prefix)

    for label in labels:
        binary_image = np.zeros(segmentation.shape, dtype=np.uint8)

        binary_image[segmentation == label] = 1

        # get the meshes from the marching cubes
        verts, faces, normals, values = measure.marching_cubes_lewiner(binary_image)
        nverts = verts.shape[0]
        nfaces = faces.shape[0]

        # center all of the vertices
        scale_factor = 1000.0
        for iv in range(nverts):
            verts[iv,0] = resolution[IB_Z] * downsample_rate[IB_Z] * (verts[iv,0] - segmentation.shape[IB_Z] / 2.0) / scale_factor
            verts[iv,1] = resolution[IB_Y] * downsample_rate[IB_Y] * (verts[iv,1] - segmentation.shape[IB_Y] / 2.0) / scale_factor
            verts[iv,2] = resolution[IB_X] * downsample_rate[IB_X] * (verts[iv,2] - segmentation.shape[IB_X] / 2.0) / scale_factor

        stl_mesh = mesh.Mesh(np.zeros(nfaces, dtype=mesh.Mesh.dtype))
        for ii, face in enumerate(faces):
            for jj in range(3):
                stl_mesh.vectors[ii][jj] = verts[face[jj],:]

        output_filename = 'meshes/{}-{}.stl'.format(prefix, label)
        stl_mesh.save(output_filename)
        print 'Wrote {}'.format(output_filename)
