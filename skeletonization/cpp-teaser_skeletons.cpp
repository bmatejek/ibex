#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <vector>
#include <queue>
#include <math.h>
#include "cpp-MinBinaryHeap.h"


// constant variables

static const int IB_Z = 0;
static const int IB_Y = 1;
static const int IB_X = 2;
static const int IB_NDIMS = 3;

// variables for the rolling sphere

static const double scale = 1.1;
static const double buffer = 80;
static const long min_path_length = 45;



// global variables

static long world_res[3];
static long grid_size[3];
static long nentries;
static long sheet_size;
static long row_size;
static double infinity;
static long inside_voxels = 0;


// global arrays

static unsigned char *skeleton = NULL;
static unsigned char *segmentation = NULL;
static double *DBF = NULL;
static double *penalties = NULL;
static double *PDRF = NULL;
static unsigned char *inside = NULL;




// helper functions

static void IndexToIndices(long iv, long &ix, long &iy, long &iz)
{
    iz = iv / sheet_size;
    iy = (iv - iz * sheet_size) / row_size;
    ix = iv % row_size;
}



static long IndicesToIndex(long ix, long iy, long iz)
{   
    return iz * sheet_size + iy * row_size + ix;
}



static void ComputeDistanceFromBoundaryField(void)
{
    // allocate memory for bounday map and distance transform
    long *b = new long[nentries];
    for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
            for (long ix = 0; ix < grid_size[IB_X]; ++ix) {
                if (!segmentation[IndicesToIndex(ix, iy, iz)]) {
                    b[IndicesToIndex(ix, iy, iz)] = 0;
                    continue;
                }

                if ((ix == 0 or iy == 0 or iz == 0 or (ix == grid_size[IB_X] - 1) or (iy == grid_size[IB_Y] - 1) or (iz == grid_size[IB_Z] - 1)) ||
                    (ix > 0 and !segmentation[IndicesToIndex(ix - 1, iy, iz)]) ||
                    (iy > 0 and !segmentation[IndicesToIndex(ix, iy - 1, iz)]) ||
                    (iz > 0 and !segmentation[IndicesToIndex(ix, iy, iz - 1)]) ||
                    (ix < grid_size[IB_X] - 1 and !segmentation[IndicesToIndex(ix + 1, iy, iz)]) ||
                    (iy < grid_size[IB_Y] - 1 and !segmentation[IndicesToIndex(ix, iy + 1, iz)]) ||
                    (iz < grid_size[IB_Z] - 1 and !segmentation[IndicesToIndex(ix, iy, iz + 1)])) {
                    b[IndicesToIndex(ix, iy, iz)] = 0;
                }
                else {
                    b[IndicesToIndex(ix, iy, iz)] = infinity;
                }
            }
        }
    }
    
    // go along the z dimenion first for every (x, y) coordinate
    for (long ix = 0; ix < grid_size[IB_X]; ++ix) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {

            long k = 0;
            long *v = new long[grid_size[IB_Z] + 1];
            double *z = new double[grid_size[IB_Z] + 1];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < grid_size[IB_Z]; ++q) {
                // label for jump statement
                zlabel:
                double s = ((b[IndicesToIndex(ix, iy, q)] + world_res[IB_Z] * world_res[IB_Z] * q * q) - (b[IndicesToIndex(ix, iy, v[k])] + world_res[IB_Z] * world_res[IB_Z] * v[k] * v[k])) / (float)(2 * world_res[IB_Z] * q - 2 * world_res[IB_Z] * v[k]);
                
                if (s <= z[k]) {
                    --k;
                    goto zlabel;
                }
                else {
                    ++k;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = infinity;
                }
            }

            k = 0;
            for (long q = 0; q < grid_size[IB_Z]; ++q) {
                while (z[k + 1] < world_res[IB_Z] * q)
                    ++k;

                DBF[IndicesToIndex(ix, iy, q)] = world_res[IB_Z] * world_res[IB_Z] * (q - v[k]) * (q - v[k]) + b[IndicesToIndex(ix, iy, v[k])];
            }

            // free memory 
            delete[] v;
            delete[] z;
        }
    }

    // update the boundary values with this distance
    for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
            for (long ix = 0; ix < grid_size[IB_X]; ++ix) {
                b[IndicesToIndex(ix, iy, iz)] = DBF[IndicesToIndex(ix, iy, iz)];
            }
        }
    }

    // go along the y dimension second for every (z, x) coordinate
    for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {
        for (long ix = 0; ix < grid_size[IB_X]; ++ix) {

            long k = 0;
            long *v = new long[grid_size[IB_Y] + 1];
            double *z = new double[grid_size[IB_Y] + 1];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < grid_size[IB_Y]; ++q) {
                // label for jump statement
                ylabel:
                double s = ((b[IndicesToIndex(ix, q, iz)] + world_res[IB_Y] * world_res[IB_Y] * q * q) - (b[IndicesToIndex(ix, v[k], iz)] +  world_res[IB_Y] * world_res[IB_Y] * v[k] * v[k])) / (float)(2 * world_res[IB_Y] * q - 2 * world_res[IB_Y] * v[k]);
                
                if (s <= z[k]) {
                    --k;
                    goto ylabel;
                }
                else {
                    ++k; 
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = infinity;
                }
            }

            k = 0;
            for (long q = 0; q < grid_size[IB_Y]; ++q) {
                while (z[k + 1] < q * world_res[IB_Y])
                    ++k;
            
                DBF[IndicesToIndex(ix, q, iz)] = world_res[IB_Y] * world_res[IB_Y] * (q - v[k]) * (q - v[k]) + b[IndicesToIndex(ix, v[k], iz)];
            }

            // free memory
            delete[] v;
            delete[] z;
        }
    }

    // update the boundary values with this distance
    for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
            for (long ix = 0; ix < grid_size[IB_X]; ++ix) {
                b[IndicesToIndex(ix, iy, iz)] = DBF[IndicesToIndex(ix, iy, iz)];
            }
        }
    }



    // go along the x dimension last for every (y, z) coordinate
    for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
        for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {

            long k = 0;
            long *v = new long[grid_size[IB_X] + 1];
            double *z = new double[grid_size[IB_X] + 1];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < grid_size[IB_X]; ++q) {
                // label for jump statement
                xlabel:
                double s = ((b[IndicesToIndex(q, iy, iz)] + world_res[IB_X] * world_res[IB_X] * q * q) - (b[IndicesToIndex(v[k], iy, iz)] +  world_res[IB_X] * world_res[IB_X] * v[k] * v[k])) / (float)(2 * world_res[IB_X] * q - 2 * world_res[IB_X] * v[k]);

                if (s <= z[k]) {
                    --k;
                    goto xlabel;
                }
                else {
                    ++k;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = infinity;
                }
            }

            k = 0;
            for (long q = 0;  q < grid_size[IB_X]; ++q) {
                while (z[k + 1] < world_res[IB_X] * q)
                    ++k;

                DBF[IndicesToIndex(q, iy, iz)] = world_res[IB_X] * world_res[IB_X] * (q - v[k]) * (q - v[k]) + b[IndicesToIndex(v[k], iy, iz)];
            }

            // free memory
            delete[] v;
            delete[] z;
        }
    }

    for (long iv = 0; iv < nentries; ++iv) {
        DBF[iv] = sqrt(DBF[iv]);
    }

    // free memory
    delete[] b;
}



struct DijkstraData {
    long iv;
    DijkstraData *prev;
    double voxel_penalty;
    double distance;
    bool visited;
};



long ComputeDistanceFromVoxelField(long source_index)
{
    DijkstraData *voxel_data = new DijkstraData[nentries];
    if (!voxel_data) exit(-1);

    // initialize all data
    for (int iv = 0; iv < nentries; ++iv) {
        voxel_data[iv].iv = iv;
        voxel_data[iv].prev = NULL;
        voxel_data[iv].voxel_penalty = penalties[iv];
        voxel_data[iv].distance = infinity;
        voxel_data[iv].visited = false;
    }

    // initialize the priority queue
    DijkstraData tmp;
    MinBinaryHeap<DijkstraData *> voxel_heap(&tmp, (&tmp.distance), nentries);

    // insert the source into the heap
    voxel_data[source_index].distance = 0.0;
    voxel_data[source_index].visited = true;
    voxel_heap.Insert(source_index, &(voxel_data[source_index]));

    // visit all vertices
    long voxel_index;
    while (!voxel_heap.IsEmpty()) {
        DijkstraData *current = voxel_heap.DeleteMin();
        voxel_index = current->iv;

        // visit all 26 neighbors of this index
        long ix, iy, iz;
        IndexToIndices(voxel_index, ix, iy, iz);

        for (long iw = iz - 1; iw <= iz + 1; ++iw) {
            if (iw < 0 or iw >= grid_size[IB_Z]) continue;
            for (long iv = iy - 1; iv <= iy + 1; ++iv) {
                if (iv < 0 or iv >= grid_size[IB_Y]) continue;
                for (long iu = ix - 1; iu <= ix + 1; ++iu) {
                    if (iu < 0 or iu >= grid_size[IB_X]) continue;
                    
                    // get the linear index for this voxel
                    long neighbor_index = IndicesToIndex(iu, iv, iw);

                    // skip if background
                    if (!segmentation[neighbor_index]) continue;
                    
                    // get the corresponding neighbor data
                    DijkstraData *neighbor_data = &(voxel_data[neighbor_index]);

                    // find the distance between these voxels
                    long deltaz = world_res[IB_Z] * (iw - iz);
                    long deltay = world_res[IB_Y] * (iv - iy);
                    long deltax = world_res[IB_X] * (iu - ix);

                    // get the distance between (ix, iy, iz) and (iu, iv, iw)
                    double distance = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);

                    // get the distance to get to this voxel through the current voxel (requires a penalty for visiting this voxel)
                    double distance_through_current = current->distance + distance + neighbor_data->voxel_penalty;
                    double distance_without_current = neighbor_data->distance;

                    if (!neighbor_data->visited) {
                        neighbor_data->prev = current;
                        neighbor_data->distance = distance_through_current;
                        neighbor_data->visited = true;
                        voxel_heap.Insert(neighbor_index, neighbor_data);
                    }
                    else if (distance_through_current < distance_without_current) {
                        neighbor_data->prev = current;
                        neighbor_data->distance = distance_through_current;
                        voxel_heap.DecreaseKey(neighbor_index, neighbor_data);
                    }
                }
            }
        }
    }


    // first call to this function needs to return the root
    if (!PDRF) {
        // free memory
        delete[] voxel_data;

        // return the farthest voxel (to get the root voxel)
        return voxel_index;
    }

    // save the PDRF (only called when given root voxel)
    for (long iv = 0; iv < nentries; ++iv) {
        if (!segmentation[iv]) continue;
        PDRF[iv] = voxel_data[iv].distance;
    }

    // continue until there are no more inside voxels
    while (inside_voxels) {
        printf("  Remaining inside voxels %d...\n", inside_voxels);
        double farthest_pdrf = -1;
        long starting_voxel = -1;

        // find the farthest PDFR that is still inside
        for (long iv = 0; iv < nentries; ++iv) {
            if (!inside[iv]) continue;

            if (PDRF[iv] > farthest_pdrf) {
                farthest_pdrf = PDRF[iv];
                starting_voxel = iv;
            }
        }

        for (long iv = 0; iv < nentries; ++iv) {
            if (!inside[iv]) continue;

            long ix, iy, iz;
            IndexToIndices(iv, ix, iy, iz);
        
            // get the skeleton path from this location to the root
            DijkstraData *current = &(voxel_data[starting_voxel]);

            while (!skeleton[current->iv]) {
                long ii, ij, ik;
                IndexToIndices(current->iv, ii, ij, ik);

                // what is the distance between this skeleton location and the inside location
                double deltax = world_res[IB_X] * (ii - ix);
                double deltay = world_res[IB_Y] * (ij - iy);
                double deltaz = world_res[IB_Z] * (ik - iz);

                double distance = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);

                if (distance < scale * DBF[current->iv] + buffer) {
                    inside[iv] = 0;
                    inside_voxels--;
                    break;
                }

                // update skeleton pointer
                current = current->prev;
            }
        }

        long skeleton_path_length = 0;
        DijkstraData *current = &(voxel_data[starting_voxel]);
        while (!skeleton[current->iv]) {
            skeleton_path_length++;
            current = current->prev;
        }

        if (skeleton_path_length > min_path_length) {
            current = &(voxel_data[starting_voxel]);
            while (current != NULL) {
                skeleton[current->iv] = 1;
                current = current->prev;
            }
        }
    }

    // free memory
    delete[] voxel_data;

    return -1;

}



void ComputePenalties(void)
{
    // get the maximum distance from the boundary
    double M = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (DBF[iv] > M) {
            M = DBF[iv];
        }
    }

    // choose 5000 so that 3000 length voxel paths have correct floating point precision
    const double pdrf_scale = 5000;
    for (long iv = 0; iv < nentries; ++iv) {
        penalties[iv] = pdrf_scale * pow(1 - DBF[iv] / M, 16);
    }
}




unsigned char *CppGenerateTeaserSkeletons(long *input_segmentation, long input_grid_size[3], long input_world_res[3])
{
    // initialize convenient variables for skeletonization
    for (int dim = 0; dim < IB_NDIMS; ++dim) {
        grid_size[dim] = input_grid_size[dim];
        world_res[dim] = input_world_res[dim];
    }
    nentries = grid_size[IB_Z] * grid_size[IB_Y] * grid_size[IB_X];
    sheet_size = grid_size[IB_Y] * grid_size[IB_X];
    row_size = grid_size[IB_X];
    infinity = (grid_size[IB_Z] * world_res[IB_Z]) * (grid_size[IB_Z] * world_res[IB_Z]) + (grid_size[IB_Y] * world_res[IB_Y]) * (grid_size[IB_Y] * world_res[IB_Y]) + (grid_size[IB_X] * world_res[IB_X]) * (grid_size[IB_X] * world_res[IB_X]);

    // create global segmentation array
    skeleton = new unsigned char[nentries];
    segmentation = new unsigned char[nentries];
    penalties = new double[nentries];
    inside = new unsigned char[nentries];

    for (long iv = 0; iv < nentries; ++iv) {
        skeleton[iv] = 0;
        segmentation[iv] = input_segmentation[iv];
        penalties[iv] = 0;
        if (segmentation[iv]) { inside[iv] = 1; inside_voxels++; }
        else { inside[iv] = 0; }
    }

    // initialize array for distance to boundary field
    DBF = new double[nentries];
    if (!DBF) exit(-1);

    clock_t t1, t2;
    t1 = clock();
    ComputeDistanceFromBoundaryField();
    t2 = clock();
    printf("Computed distance from boundary field in %lf seconds\n", ((float)t2 - (float)t1) / CLOCKS_PER_SEC);


    t1 = clock();
    long source_voxel = -1;
    for (long iv = 0; iv < nentries; ++iv)
        if (inside[iv]) { source_voxel = iv; break; }
    if (source_voxel == -1) { 
        fprintf(stderr, "No voxels with this label...\n"); 
        delete[] DBF;
        delete[] penalties;
        delete[] segmentation;
        delete[] inside;
        return skeleton;
    }
    long root_voxel = ComputeDistanceFromVoxelField(source_voxel);
    // the root location starts the skeleton
    skeleton[root_voxel] = 1;
    inside[root_voxel] = 0;
    inside_voxels--;


    t2 = clock();
    printf("Computed distance from any voxel field in %lf seconds\n", ((float)t2 - (float)t1) / CLOCKS_PER_SEC);

    t1 = clock();
    ComputePenalties();
    // initialize here so previous call to ComputeDistanceFromVoxelField returns root voxel
    PDRF = new double[nentries];
    if (!PDRF) exit(-1);
    ComputeDistanceFromVoxelField(root_voxel);
    t2 = clock();
    printf("Computed penalized distance from root voxel field in %lf seconds\n", ((float)t2 - (float)t1) / CLOCKS_PER_SEC);

    // free memory
    delete[] DBF;
    delete[] penalties;
    delete[] PDRF;
    delete[] segmentation;
    delete[] inside;

    // reset global variables
    DBF = NULL;
    penalties = NULL;
    PDRF = NULL;
    segmentation = NULL;
    inside = NULL;

    return skeleton;
}