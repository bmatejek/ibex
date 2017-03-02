#include <stdio.h>
#include "cpp-skeletonize.h"



// data size variables
static unsigned long zres;
static unsigned long yres;
static unsigned long xres;
static unsigned long grid_size;
static unsigned long sheet_size;
static unsigned long row_size;
// dummy variable for large numbers
static unsigned long infinity;



int IndicesToIndex(int ix, int iy, int iz)
{
    return iz * sheet_size + iy * row_size + ix;
}



void IndexToIndices(int index, int& ix, int& iy, int& iz)
{
    iz = index / sheet_size;
    iy = (index - iz * sheet_size) / row_size;
    ix = index % row_size;
}



unsigned long *DistanceTransform(unsigned long *segmentation) 
{
    // create an empty array of distances
    unsigned long *dt = new unsigned long[grid_size];
    unsigned long *b = new unsigned long[grid_size];
    for (unsigned int iv = 0; iv < grid_size; ++iv) {
        b[iv] = infinity;

        // get the current index
         int ix;
         int iy;
         int iz;
        IndexToIndices(iv, ix, iy, iz);

        // check the north, south, east, west neighbors
        if (ix > 0) {
            int north_index = IndicesToIndex(ix - 1, iy, iz);
            if (segmentation[north_index] != segmentation[iv]) b[iv] = 0;
        }
        if (ix < xres - 1) {
            int south_index = IndicesToIndex(ix + 1, iy, iz);
            if (segmentation[south_index] != segmentation[iv]) b[iv] = 0;
        }
        if (iy < yres - 1) {
            int east_index = IndicesToIndex(ix, iy + 1, iz);
            if (segmentation[east_index] != segmentation[iv]) b[iv] = 0;
        }
        if (iy > 0) {
            int west_index = IndicesToIndex(ix, iy - 1, iz);
            if (segmentation[west_index] != segmentation[iv]) b[iv] = 0;
        }
    }

    // iterate over every slice
    for (unsigned int iz = 0; iz < zres; ++iz) {
        // get the distance transform along y
        for (unsigned int ix = 0; ix < xres; ++ix) {
            int k = 0;
            int *v = new int[yres];
            float *z = new float[yres];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (unsigned int q = 1; q < yres; ++q) {
            ylabel:
                float s = ((b[IndicesToIndex(ix, q, iz)] + q * q) - (b[IndicesToIndex(ix, v[k], iz)] + v[k] * v[k])) / (float)(2 * q - 2 * v[k]);
                if (s <= z[k]) {
                    k = k - 1;
                    goto ylabel;
                }
                else {
                    k = k + 1;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = infinity;
                }
            }

            k = 0;
            for (unsigned int q = 0; q < yres; ++q) {
                while (z[k + 1] < q) {
                    k = k + 1;
                }

                dt[IndicesToIndex(ix, q, iz)] = (q - v[k]) * (q - v[k]) + b[IndicesToIndex(ix, v[k], iz)];
            }

            delete[] v;
            delete[] z;
        }
            
        for (int iy = 0; iy < yres; ++iy) {
            for (int ix = 0; ix < xres; ++ix) {
                int iv = IndicesToIndex(ix, iy, iz);
                b[iv] = dt[iv];
            }
        }

        // get the distance transform along x
        for (unsigned int iy = 0; iy < yres; ++iy) {
            int k = 0; 
            int *v = new int[xres];
            float *z = new float[xres];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (unsigned int q = 1; q < xres; ++q) {
            xlabel:
                float s = ((b[IndicesToIndex(q, iy, iz)] + q * q) - (b[IndicesToIndex(v[k], iy, iz)] + v[k] * v[k])) / (float)(2 * q - 2 * v[k]);
                if (s <= z[k]) {
                    k = k - 1;
                    goto xlabel;
                }
                else {
                    k = k + 1;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = infinity;
                }
            }

            k = 0;
            for (unsigned int q = 0; q < xres; ++q) {
                while (z[k + 1] < q) {
                    k = k + 1;
                }

                dt[IndicesToIndex(q, iy, iz)] = (q - v[k]) * (q - v[k]) + b[IndicesToIndex(v[k], iy, iz)];
            }

            delete[] v;
            delete[] z;
        }
    }

    // return the distance transform
    return dt;
}



unsigned long *Skeletonize(unsigned long *segmentation, int input_zres, int input_yres, int input_xres)
{
    // update global variables
    zres = input_zres;
    yres = input_yres; 
    xres = input_xres;

    grid_size = zres * yres * xres;
    sheet_size = yres * xres;
    row_size = xres;

    infinity = grid_size * grid_size * 3;

    // compute two dimensional distance transform over every slice
    unsigned long *dt = DistanceTransform(segmentation);

    return dt;
}