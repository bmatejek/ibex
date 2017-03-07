#include <stdio.h>
#include <ctime>
#include "cpp-skeletonize.h"
#include <vector>
#include <queue>



// data size variables
static long zres;
static long yres;
static long xres;
static long grid_size;
static long sheet_size;
static long row_size;

// dummy variable for large numbers
static long infinity;



////////////////////////////////
//// VOXEL ACCESS FUNCTIONS ////
////////////////////////////////

enum TWO_DIMENSIONAL_DIRECTION { NORTH_2D, EAST_2D, SOUTH_2D, WEST_2D, N_2D_DIRECTIONS };
enum THREE_DIMENSIONAL_DIRECTION { NORTH_3D, EAST_3D, SOUTH_3D, WEST_3D, UP_3D, DOWN_3D, N_3D_DIRECTIONS};



inline long IndicesToIndex(long ix, long iy, long iz)
{
    return iz * sheet_size + iy * row_size + ix;
}



inline void IndexToIndices(long index, long& ix, long& iy, long& iz)
{
    iz = index / sheet_size;
    iy = (index - iz * sheet_size) / row_size;
    ix = index % row_size;
}



long TwoDimensionalNeighbor(long iv, TWO_DIMENSIONAL_DIRECTION direction)
{
    long ix, iy, iz;
    IndexToIndices(iv, ix, iy, iz);

    if (direction == NORTH_2D) {
        if (ix == 0) return -1;
        else return IndicesToIndex(ix - 1, iy, iz);
    }
    if (direction == EAST_2D) {
        if (iy == yres - 1) return -1;
        else return IndicesToIndex(ix, iy + 1, iz);
    }
    if (direction == SOUTH_2D) {
        if (ix == xres - 1) return -1;
        else return IndicesToIndex(ix + 1, iy, iz);
    }
    if (direction == WEST_2D) {
        if (iy == 0) return -1;
        else return IndicesToIndex(ix, iy - 1, iz);
    }

    return -1;
}



long ThreeDimensionalNeighbor(long iv, THREE_DIMENSIONAL_DIRECTION direction)
{
    long ix, iy, iz;
    IndexToIndices(iv, ix, iy, iz);

    if (direction == NORTH_3D) {
        if (ix == 0) return -1;
        else return IndicesToIndex(ix - 1, iy, iz);
    }
    if (direction == EAST_3D) {
        if (iy == yres - 1) return -1;
        else return IndicesToIndex(ix, iy + 1, iz);
    }
    if (direction == SOUTH_3D) {
        if (ix == xres - 1) return -1;
        else return IndicesToIndex(ix + 1, iy, iz);
    }
    if (direction == WEST_3D) {
        if (iy == 0) return -1;
        else return IndicesToIndex(ix, iy - 1, iz);
    }
    if (direction == UP_3D) {
        if (iz == zres - 1) return -1;
        else return IndicesToIndex(ix, iy, iz + 1);
    }
    if (direction == DOWN_3D) {
        if (iz == 0) return -1;
        else return IndicesToIndex(ix, iy, iz - 1);
    }

    return -1;
}



//////////////////////////////////////
//// DISTANCE TRANSFORM FUNCTIONS ////
//////////////////////////////////////

long *TwoDimensionalDistanceTransform(unsigned long *segmentation, long *boundaries)
{
    // allocate memory for boundary map and distance transform
    long *dt = new long[grid_size];
    long *b = new long[grid_size];
    for (int iv = 0; iv < grid_size; ++iv)
        b[iv] = boundaries[iv];

    for (long iz = 0; iz < zres; ++iz) {
        // run along all y scanlines
        for (long iy = 0; iy < yres; ++iy) {
            long k = 0;
            long *v = new long[xres];
            double *z = new double[xres];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < xres; ++q) {
                xlabel:

                double s = ((b[IndicesToIndex(q, iy, iz)] + q * q) - (b[IndicesToIndex(v[k], iy, iz)] + v[k] * v[k])) / (float)(2 * q - 2 * v[k]);
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
            for (long q = 0; q < xres; ++q) {
                while (z[k + 1] < q) 
                    k = k + 1;

                dt[IndicesToIndex(q, iy, iz)] = (q - v[k]) * (q - v[k]) + b[IndicesToIndex(v[k], iy, iz)];
            }

            // free memory
            delete[] v;
            delete[] z;
        }

        for (int iy = 0; iy < yres; ++iy) {
            for (int ix = 0; ix < xres; ++ix) {
                int iv = IndicesToIndex(ix, iy, iz);
                b[iv] = dt[iv];
            }
        }

        // run along all x scanlines
        for (long ix = 0; ix < xres; ++ix) {
            long k = 0;
            long *v = new long[yres];
            double *z = new double[yres];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < yres; ++q) {
                ylabel:

                double s = ((b[IndicesToIndex(ix, q, iz)] + q * q) - (b[IndicesToIndex(ix, v[k], iz)] + v[k] * v[k])) / (float)(2 * q - 2 * v[k]);
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
            for (long q = 0; q < yres; ++q) {
                while (z[k + 1] < q)
                    k = k + 1;

                dt[IndicesToIndex(ix, q, iz)] = (q - v[k]) * (q - v[k]) + b[IndicesToIndex(ix, v[k], iz)];
            }

            // free memory
            delete[] v;
            delete[] z;
        }
    }

    // free memory
    delete[] b;

    return dt;
}



/////////////////////////////
//// DIJKSTTRA ALGORITHM ////
/////////////////////////////

struct DijkstraVoxelNode
{
    long index;
    DijkstraVoxelNode *previous;
    long distance;
    bool visited;
};



class Compare {
public:
    bool operator() (DijkstraVoxelNode *a, DijkstraVoxelNode *b)
    {
        return a->distance > b->distance;
    }
};



void *DijkstraAlgorithm(std::vector<long> &sources, long *boundaries)
{
    // allocate temporary data
    DijkstraVoxelNode *voxel_data = new DijkstraVoxelNode[grid_size];
    if (!voxel_data) { fprintf(stderr, "Failed to allocate temporary data for geodisic distances\n"); return 0; }

    for (long iv = 0; iv < grid_size; ++iv) {
        voxel_data[iv].index = iv;
        voxel_data[iv].previous = NULL;
        voxel_data[iv].distance = infinity;
        voxel_data[iv].visited = false;
    }


    std::priority_queue<DijkstraVoxelNode *, std::vector<DijkstraVoxelNode *>, Compare> heap;    

    for (unsigned int is = 0; is < sources.size(); ++is) {
        voxel_data[is].distance = 0;
        voxel_data[is].visited = true;
        heap.push(&(voxel_data[is]));
    }

    while (!heap.empty()) {
        // pop off the top element
        DijkstraVoxelNode *current = heap.top();
        heap.pop();

        // go through all of the neighbors
        int index = current->index;
        for (int in = 0; in < N_3D_DIRECTIONS; ++in) {
            int neighbor_index = 
        }

    }

}



unsigned long *Skeletonize(unsigned long *segmentation, int input_zres, int input_yres, int input_xres)
{
    ///////////////////////
    //// PREPROCESSING ////
    ///////////////////////

    // update global variables
    zres = input_zres;
    yres = input_yres; 
    xres = input_xres;

    grid_size = zres * yres * xres;
    sheet_size = yres * xres;
    row_size = xres;

    infinity = xres * xres + yres * yres + zres * zres;

    // create the boundary map
    long *boundaries = new long[grid_size];
    for (long iv = 0; iv < grid_size; ++iv) {
        unsigned long label = segmentation[iv];
        
        // consider all neighbors
        bool interior = true;
        for (int in = 0; in < N_2D_DIRECTIONS; ++in) {
            long neighbor_index = TwoDimensionalNeighbor(iv, TWO_DIMENSIONAL_DIRECTION(in));    
            if (neighbor_index == -1) continue;

            // not interior if neighbor disagrees
            if (segmentation[neighbor_index] != label) interior = false;
        }

        // update the boundary map
        if (interior) boundaries[iv] = infinity;
        else boundaries[iv] = 0;
    }



    //////////////////
    //// STEP ONE ////
    //////////////////

    std::time_t start_time = std::time(NULL);
    long *dt = TwoDimensionalDistanceTransform(segmentation, boundaries);

    // find the max segmentation value
    unsigned long max_segmentation = 0;
    for (long iv = 0; iv < grid_size; ++iv) {
        if (segmentation[iv] > max_segmentation) max_segmentation = segmentation[iv];
    }

    // find the location for each segment that has the largest distance transform value
    long *argmax_dt = new long[max_segmentation + 1];
    for (unsigned long iv = 0; iv < max_segmentation; ++iv)
        argmax_dt[iv] = -1;

    // iterate over the entire volume
    for (long iv = 0; iv < grid_size; ++iv) {
        unsigned long label = segmentation[iv];
        long distance = dt[iv];

        if (distance > argmax_dt[label]) argmax_dt[label] = iv;
    }
    printf("First step completed in %lu seconds\n", std::time(NULL) - start_time);



    //////////////////
    //// STEP TWO ////
    //////////////////

    start_time = std::time(NULL);


    printf("Second step completed in %lu seconds\n", std::time(NULL) - start_time);

    return (unsigned long *) boundaries;
}