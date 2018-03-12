import numpy as np

from skimage import measure
from stl import mesh
from numba import jit


from ibex.utilities.constants import *
from ibex.utilities import dataIO
from ibex.transforms import h52h5




# run the marching cube algorithm for these labels
def MarchingCubes(data, resolution, downsample_rate, output_prefix):
    # downsample the input data to a manageable size
    downsampled_data = h52h5.DownsampleData(data, downsample_rate)

    # get the new grid size
    zres, yres, xres = downsampled_data.shape

    # go through all labels
    labels = np.unique(downsampled_data)
    # to make sure masking occurred (arbitrary value)
    assert (len(labels) < 100)

    for label in labels:
        if not label: continue

        # create a binary image for this label
        binary_image = np.zeros(downsampled_data.shape, dtype=np.uint8)
        binary_image[downsampled_data == label] = 1

        # get the meshes from the marching cubes
        verts, faces, normals, values = measure.marching_cubes_lewiner(binary_image)
        nverts = verts.shape[0]
        nfaces = faces.shape[0]

        # center all of the vertices
        scale_factor = 1000.0
        for iv in range(nverts):
            # divide by 2.0 to center the vertices
            verts[iv,0] = resolution[IB_Z] * downsample_rate[IB_Z] * (verts[iv,IB_Z] - zres / 2.0) / scale_factor
            verts[iv,1] = resolution[IB_Y] * downsample_rate[IB_Y] * (verts[iv,IB_Y] - yres / 2.0) / scale_factor
            verts[iv,2] = resolution[IB_X] * downsample_rate[IB_X] * (verts[iv,IB_X] - xres / 2.0) / scale_factor

        stl_mesh = mesh.Mesh(np.zeros(nfaces, dtype=mesh.Mesh.dtype))
        for ii, face in enumerate(faces):
            for jj in range(3):
                stl_mesh.vectors[ii][jj] = verts[face[jj],:]

        output_filename = '{}-{}.stl'.format(output_prefix, label)
        stl_mesh.save(output_filename)
        print 'Wrote {}'.format(output_filename)
