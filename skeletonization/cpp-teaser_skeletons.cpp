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

static const double scale = 50.0;
static const double buffer = 15000.0;



// global variables

static long world_res[3];
static long grid_size[3];
static long nentries;
static long sheet_size;
static long row_size;
static long infinity;

static long inside_voxels_remaining = 0;
static unsigned char *skeleton = NULL;
static unsigned char *inside = NULL;
static long *segmentation = NULL;
static double *DBF = NULL;
static double *penalties = NULL;
static double *PDRF = NULL;




// helper functions

static void IndexToIndicies(long iv, long &ix, long &iy, long &iz)
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
                long label = segmentation[IndicesToIndex(ix, iy, iz)];

                if ((ix == 0 or iy == 0 or iz == 0 or (ix == grid_size[IB_X] - 1) or (iy == grid_size[IB_Y] - 1) or (iz == grid_size[IB_Z] - 1)) ||
                    (ix > 0 and segmentation[IndicesToIndex(ix - 1, iy, iz)] != label) ||
                    (iy > 0 and segmentation[IndicesToIndex(ix, iy - 1, iz)] != label) ||
                    (iz > 0 and segmentation[IndicesToIndex(ix, iy, iz - 1)] != label) ||
                    (ix < grid_size[IB_X] - 1 and segmentation[IndicesToIndex(ix + 1, iy, iz)] != label) ||
                    (iy < grid_size[IB_Y] - 1 and segmentation[IndicesToIndex(ix, iy + 1, iz)] != label) ||
                    (iz < grid_size[IB_Z] - 1 and segmentation[IndicesToIndex(ix, iy, iz + 1)] != label)) {
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
            long *v = new long[grid_size[IB_Z]];
            double *z = new double[grid_size[IB_Z]];

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
            long *v = new long[grid_size[IB_Y]];
            double *z = new double[grid_size[IB_Y]];

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
            long *v = new long[grid_size[IB_X]];
            double *z = new double[grid_size[IB_X]];

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

long ComputeDistanceFromVoxelField(long source_index, bool calculate_pdrf, bool skeletonize)
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

        // stop when the spine is reached
        if (skeleton[voxel_index]) break;

        // visit all 26 neighbors of this index
        long ix, iy, iz;
        IndexToIndicies(voxel_index, ix, iy, iz);

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
                    DijkstraData *neighbor_data = &(voxel_data[neighbor_index]);

                    // find the distance between these voxels
                    long deltaz = world_res[IB_Z] * (iw - iz);
                    long deltay = world_res[IB_Y] * (iv - iy);
                    long deltax = world_res[IB_X] * (iu - ix);

                    double distance = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);

                    // get the distance to get to this voxel through the current voxel (requires a penalty for visiting this voxel)
                    double distance_through_current = current->distance + distance + current->voxel_penalty;
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

    // skeletonize from the starting voxel to the current skeleton
    if (skeletonize) {
        DijkstraData *current = &(voxel_data[voxel_index]);

        // save the skeleton
        while (current != NULL) {
            skeleton[current->iv] = 1;
            current = current->prev;
        }

        // mask out voxels near the skeleton
        for (long iv = 0; iv < nentries; ++iv) {
            if (not inside[iv]) continue;
            long ix, iy, iz;
            IndexToIndicies(iv, ix, iy, iz);

            DijkstraData *current = &(voxel_data[voxel_index]);

            // save the skeleton
            while (current != NULL) {
                long skeleton_index = current->iv;

                long ii, ij, ik;
                IndexToIndicies(skeleton_index, ii, ij, ik);

                double distance = world_res[IB_X] * world_res[IB_X] * (ii - ix) * (ii - ix) + world_res[IB_Y] * world_res[IB_Y] * (ij - iy) * (ij - iy) + world_res[IB_Z] * world_res[IB_Z] * (ik - iz) * (ik - iz);

                if (distance < scale * DBF[skeleton_index] + buffer) {
                    inside[iv] = 0;
                    inside_voxels_remaining--;
                    break;
                }

                current = current->prev;
            }
        }
    }

    // save the PDRF (only called when given root voxel)
    if (calculate_pdrf) {
        for (long iv = 0; iv < nentries; ++iv) 
            PDRF[iv] = voxel_data[iv].distance;
    }

    // free memory
    delete[] voxel_data;

    // return the farthest voxel (briefly needed)
    return voxel_index;
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




unsigned char *CppGenerateTeaserSkeletons(long *input_segmentation, long label, long input_grid_size[3], long input_world_res[3])
{
    // initialize convenient variables for skeletons
    for (int dim = 0; dim < IB_NDIMS; ++dim) {
        grid_size[dim] = input_grid_size[dim];
        world_res[dim] = input_world_res[dim];
    }
    nentries = grid_size[IB_Z] * grid_size[IB_Y] * grid_size[IB_X];
    sheet_size = grid_size[IB_Y] * grid_size[IB_X];
    row_size = grid_size[IB_X];
    infinity = (grid_size[IB_Z] * world_res[IB_Z]) * (grid_size[IB_Z] * world_res[IB_Z]) + (grid_size[IB_Y] * world_res[IB_Y]) * (grid_size[IB_Y] * world_res[IB_Y]) + (grid_size[IB_X] * world_res[IB_X]) * (grid_size[IB_X] * world_res[IB_X]);

    // set point for segmentation
    segmentation = new long[nentries];
    for (long iv = 0; iv < nentries; ++iv)
        segmentation[iv] = input_segmentation[iv];

    for (long iv = 0; iv < nentries; ++iv) {
        if (segmentation[iv] != label) segmentation[iv] = 0;
        else { 
            segmentation[iv] = 1;
            inside_voxels_remaining++;
        }
    }

    // initialize useful arrays
    DBF = new double[nentries];
    if (!DBF) exit(-1);
    penalties = new double[nentries];
    if (!penalties) exit(-1);
    skeleton = new unsigned char[nentries];
    if (!skeleton) exit(-1);
    inside = new unsigned char[nentries];
    if (!inside) exit(-1);
    PDRF = new double[nentries];
    if (!PDRF) exit(-1);

    // initialize values to zero
    for (long iv = 0; iv < nentries; ++iv) {
        DBF[iv] = 0;
        penalties[iv] = 0;
        skeleton[iv] = 0;
        inside[iv] = segmentation[iv];
        PDRF[iv] = 0;
    }


    clock_t t1, t2;
    t1 = clock();
    // compute the distance boundary field
    ComputeDistanceFromBoundaryField();
    t2 = clock();
    printf("\n running time: %lf\n", ((float)t2 - (float)t1) / CLOCKS_PER_SEC);

    t1 = clock();
    // comput the distance from any voxel field (choose first voxel)
    long source_voxel = -1;
    for (long iv = 0; iv < nentries; ++iv) {
        if (segmentation[iv]) { source_voxel = iv; break; }
    }
    long root_voxel = ComputeDistanceFromVoxelField(source_voxel, false, false);
    t2 = clock();
    printf("\n running time: %lf\n", ((float)t2 - (float)t1) / CLOCKS_PER_SEC);

    t1 = clock();
    // calculate penalties
    ComputePenalties();
    t2 = clock();
    printf("\n running time: %lf\n", ((float)t2 - (float)t1) / CLOCKS_PER_SEC);

    t1 = clock();
    // compute penalized distance from root voxel field
    long starting_voxel = ComputeDistanceFromVoxelField(root_voxel, true, false);
    t2 = clock();
    printf("\n running time: %lf\n", ((float)t2 - (float)t1) / CLOCKS_PER_SEC);
    
    // set the root voxel, every dijkstra iteration terminates at root
    // this earlier will cause problems since the algorithm will terminate immediately
    skeleton[root_voxel] = 1;

    long index = 0;
    while (inside_voxels_remaining > 0) {
        t1 = clock();
        // find the farthest point from this starting value
        ComputeDistanceFromVoxelField(starting_voxel, false, true);
        t2 = clock();
        printf("\n running time: %lf\n", ((float)t2 - (float)t1) / CLOCKS_PER_SEC);    

        // recompute the starting voxel
        double maximum_distance = 0.0;
        for (long iv = 0; iv < nentries; ++iv) {
            if (inside[iv] and (PDRF[iv] > maximum_distance)) {
                maximum_distance = PDRF[iv];
                starting_voxel = iv;
            }
        }

        printf("Starting Voxel: %d\n", starting_voxel);
        printf("Inside Voxels: %d\n", inside_voxels_remaining);




        long actual_inside_voxels_remaining = 0;
        for (long iv = 0; iv < nentries; ++iv) {
            if (inside[iv]) actual_inside_voxels_remaining++;
        }

        if (actual_inside_voxels_remaining !=inside_voxels_remaining) {
            printf("Mismatch: %d %d\n", actual_inside_voxels_remaining, inside_voxels_remaining);
            exit(-1);
        }

        for (long iv = 0; iv < nentries; ++iv) {
            if (inside[iv] && skeleton[iv]) {
                printf("Error skeleton labeled as inside: %d\n", iv);
                exit(-1);
            }
        }
    }

    delete[] DBF;
    delete[] penalties;
    delete[] inside;
    delete[] PDRF;
    delete[] segmentation;
    
    return skeleton;
}