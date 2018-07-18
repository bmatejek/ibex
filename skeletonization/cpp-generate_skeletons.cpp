#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <set>
#include <time.h>
#include <math.h>
#include <set>
#include <assert.h>
#include "cpp-MinBinaryHeap.h"


// constant variables

static const int lookup_table_size = 1 << 23;
static const int NTHINNING_DIRECTIONS = 6;
static const int UP = 0;
static const int DOWN = 1;
static const int NORTH = 2;
static const int SOUTH = 3;
static const int EAST = 4;
static const int WEST = 5;
static const int IB_Z = 0;
static const int IB_Y = 1;
static const int IB_X = 2;
static const int IB_NDIMS = 3;



static int print_verbose = 0;



// mask variables for bitwise operations

static long int long_mask[26];
static unsigned char char_mask[8];

static void set_long_mask(void)
{
    long_mask[ 0] = 0x00000001;
    long_mask[ 1] = 0x00000002;
    long_mask[ 2] = 0x00000004;
    long_mask[ 3] = 0x00000008;
    long_mask[ 4] = 0x00000010;
    long_mask[ 5] = 0x00000020;
    long_mask[ 6] = 0x00000040;
    long_mask[ 7] = 0x00000080;
    long_mask[ 8] = 0x00000100;
    long_mask[ 9] = 0x00000200;
    long_mask[10] = 0x00000400;
    long_mask[11] = 0x00000800;
    long_mask[12] = 0x00001000;
    long_mask[13] = 0x00002000;
    long_mask[14] = 0x00004000;
    long_mask[15] = 0x00008000;
    long_mask[16] = 0x00010000;
    long_mask[17] = 0x00020000;
    long_mask[18] = 0x00040000;
    long_mask[19] = 0x00080000;
    long_mask[20] = 0x00100000;
    long_mask[21] = 0x00200000;
    long_mask[22] = 0x00400000;
    long_mask[23] = 0x00800000;
    long_mask[24] = 0x01000000;
    long_mask[25] = 0x02000000;
}

static void set_char_mask(void)
{
    char_mask[0] = 0x01;
    char_mask[1] = 0x02;
    char_mask[2] = 0x04;
    char_mask[3] = 0x08;
    char_mask[4] = 0x10;
    char_mask[5] = 0x20;
    char_mask[6] = 0x40;
    char_mask[7] = 0x80;
}



// global variables for both algorithms

static long zres;
static long yres;
static long xres;
static long nentries;
static long sheet_size;
static long row_size;
static unsigned char *segmentation = NULL;



// lookup tables

static unsigned char *lut_simple;
static unsigned char *lut_isthmus;


// global variables for TEASER algorithm

static long infinity;
static unsigned char *skeleton = NULL;
static double *DBF = NULL;
static double *penalties = NULL;
static double *PDRF = NULL;
static unsigned char *inside = NULL;
static long inside_voxels = 0;
static const double scale = 1.1;
static const double buffer = 2;
static const long min_path_length = 2;



// very simple double linked list data structure

typedef struct {
    long iv, ix, iy, iz;
    void *next;
    void *prev;
} ListElement;

typedef struct {
    void *first;
    void *last;
} List;

typedef struct {
    long iv, ix, iy, iz;
} Voxel;

typedef struct {
    Voxel v;
    ListElement *ptr;
    void *next;
} Cell;

typedef struct {
    Cell *head;
    Cell *tail;
    int length;
} PointList;

typedef struct {
    ListElement *first;
    ListElement *last;
} DoubleList;


List surface_voxels;

static void NewSurfaceVoxel(long iv, long ix, long iy, long iz)
{
    ListElement *LE = new ListElement();
    LE->iv = iv;
    LE->ix = ix;
    LE->iy = iy;
    LE->iz = iz;

    LE->next = NULL;
    LE->prev = surface_voxels.last;

    if (surface_voxels.last != NULL) ((ListElement *) surface_voxels.last)->next = LE;
    surface_voxels.last = LE;
    if (surface_voxels.first == NULL) surface_voxels.first = LE;
}

static void RemoveSurfaceVoxel(ListElement *LE) {
    ListElement *LE2;
    if (surface_voxels.first == LE) surface_voxels.first = LE->next;
    if (surface_voxels.last == LE) surface_voxels.last = LE->prev;

    if (LE->next != NULL) {
        LE2 = (ListElement *)(LE->next);
        LE2->prev = LE->prev;
    }
    if (LE->prev != NULL) {
        LE2 = (ListElement *)(LE->prev);
        LE2->next = LE->next;
    }
    delete LE;
}

static void CreatePointList(PointList *s) {
    s->head = NULL;
    s->tail = NULL;
    s->length = 0;
}

static void AddToList(PointList *s, Voxel e, ListElement *ptr) {
    Cell *newcell = new Cell();
    newcell->v = e;
    newcell->ptr = ptr;
    newcell->next = NULL;

    if (s->head == NULL) {
        s->head = newcell;
        s->tail = newcell;
        s->length = 1;
    }
    else {
        s->tail->next = newcell;
        s->tail = newcell;
        s->length++;
    }
}

static Voxel GetFromList(PointList *s, ListElement **ptr)
{
    Voxel V;
    Cell *tmp;
    V.iv = -1;
    V.ix = -1;
    V.iy = -1;
    V.iz = -1;
    (*ptr) = NULL;
    if (s->length == 0) return V;
    else {
        V = s->head->v;
        (*ptr) = s->head->ptr;
        tmp = (Cell *) s->head->next;
        delete s->head;
        s->head = tmp;
        s->length--;
        if (s->length == 0) {
            s->head = NULL;
            s->tail = NULL;
        }
        return V;
    }
}

static void DestroyPointList(PointList *s) {
    ListElement *ptr;
    while (s->length) GetFromList(s, &ptr);
}



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



static void InitializeLookupTables(const char *lookup_table_directory)
{
    char lut_filename[4096];
    FILE *lut_file;

    // read the simple lookup table
    sprintf(lut_filename, "%s/lut_simple.dat", lookup_table_directory);
    lut_simple = new unsigned char[lookup_table_size];
    lut_file = fopen(lut_filename, "rb");
    if (!lut_file) {
        fprintf(stderr, "Failed to read %s\n", lut_filename);
        exit(-1);
    }
    if (fread(lut_simple, 1, lookup_table_size, lut_file) != lookup_table_size) {
        fprintf(stderr, "Failed to read %s\n", lut_filename);
        exit(-1);
    }
    fclose(lut_file);

    // read the isthmus lookup table
    sprintf(lut_filename, "%s/lut_isthmus.dat", lookup_table_directory);
    lut_isthmus = new unsigned char[lookup_table_size];
    lut_file = fopen(lut_filename, "rb");
    if (!lut_file) {
        fprintf(stderr, "Failed to read %s\n", lut_filename);
        exit(-1);
    }
    if (fread(lut_isthmus, 1, lookup_table_size, lut_file) != lookup_table_size) {
        fprintf(stderr, "Failed to read %s\n", lut_filename);
        exit(-1);
    }
    fclose(lut_file);

    // set the mask variables
    set_char_mask();
    set_long_mask();
}



static void CollectSurfaceVoxels(void)
{
    for (long iz = 1; iz < zres - 1; ++iz) {
        for (long iy = 1; iy < yres - 1; ++iy) {
            for (long ix = 1; ix < xres - 1; ++ix) {
                long iv = IndicesToIndex(ix, iy, iz);
                if (segmentation[iv]) {
                    if (!segmentation[IndicesToIndex(ix, iy, iz - 1)] ||
                            !segmentation[IndicesToIndex(ix, iy, iz + 1)] ||
                            !segmentation[IndicesToIndex(ix, iy - 1, iz)] ||
                            !segmentation[IndicesToIndex(ix, iy + 1, iz)] ||
                            !segmentation[IndicesToIndex(ix - 1, iy, iz)] ||
                            !segmentation[IndicesToIndex(ix + 1, iy, iz)])
                    {
                        segmentation[iv] = 2;
                        NewSurfaceVoxel(iv, ix, iy, iz);
                    }
                }
            }
        }
    }
}



static unsigned int Collect26Neighbors(long ix, long iy, long iz)
{
    unsigned int neighbors = 0;
    long index = 0;
    for (long iw = iz - 1; iw <= iz + 1; ++iw) {
        for (long iv = iy - 1; iv <= iy + 1; ++iv) {
            for (long iu = ix - 1; iu <= ix + 1; ++iu) {
                if (iw == iz and iv == iy and iu == ix) continue;

                // if this has the same value the neighbor bit is 1
                if (segmentation[IndicesToIndex(iu, iv, iw)]) {
                    neighbors |= long_mask[index];
                }

                // index is out here so indices go from 0 to 26 (not 27)
                ++index;
            }
        }
    }

    return neighbors;
}



static bool Simple26_6(unsigned int neighbors)
{
    return lut_simple[(neighbors >> 3)] & char_mask[neighbors % 8];
}



static bool Isthmus(unsigned int neighbors)
{
    return lut_isthmus[(neighbors >> 3)] & char_mask[neighbors % 8];
}



static void DetectSimpleBorderPoints(PointList *deletable_points, int direction)
{
    ListElement *LE = (ListElement *)surface_voxels.first;
    while (LE != NULL) {
        long iv = LE->iv;
        long ix = LE->ix;
        long iy = LE->iy;
        long iz = LE->iz;

        // not an isthmus
        if (segmentation[iv] == 2) {
            long value = 0;
            switch (direction) {
            case UP: {
                value = segmentation[IndicesToIndex(ix, iy - 1, iz)];
                break;
            }
            case DOWN: {
                value = segmentation[IndicesToIndex(ix, iy + 1, iz)];
                break;
            }
            case NORTH: {
                value = segmentation[IndicesToIndex(ix, iy, iz - 1)];
                break;
            }
            case SOUTH: {
                value = segmentation[IndicesToIndex(ix, iy, iz + 1)];
                break;
            }
            case EAST: {
                value = segmentation[IndicesToIndex(ix + 1, iy, iz)];
                break;
            }
            case WEST: {
                value = segmentation[IndicesToIndex(ix - 1, iy, iz)];
                break;
            }
            }

            // see if the required point belongs to a different segment
            if (!value) {
                unsigned int neighbors = Collect26Neighbors(ix, iy, iz);

                // deletable point
                if (Simple26_6(neighbors)) {
                    Voxel voxel;
                    voxel.iv = iv;
                    voxel.ix = ix;
                    voxel.iy = iy;
                    voxel.iz = iz;
                    AddToList(deletable_points, voxel, LE);
                }
                else {
                    if (Isthmus(neighbors)) {
                        segmentation[iv] = 3;
                    }
                }
            }
        }
        LE = (ListElement *) LE->next;
    }
}



static long ThinningIterationStep(void)
{
    long changed = 0;

    // iterate through every direction
    for (int direction = 0; direction < NTHINNING_DIRECTIONS; ++direction) {
        PointList deletable_points;
        ListElement *ptr;

        CreatePointList(&deletable_points);
        DetectSimpleBorderPoints(&deletable_points, direction);

        while (deletable_points.length) {
            Voxel voxel = GetFromList(&deletable_points, &ptr);

            long iv = voxel.iv;
            long ix = voxel.ix;
            long iy = voxel.iy;
            long iz = voxel.iz;

            unsigned int neighbors = Collect26Neighbors(ix, iy, iz);
            if (Simple26_6(neighbors)) {
                // delete the simple point
                segmentation[iv] = 0;

                // add the new surface voxels
                if (segmentation[IndicesToIndex(ix - 1, iy, iz)] == 1) {
                    NewSurfaceVoxel(IndicesToIndex(ix - 1, iy, iz), ix - 1, iy, iz);
                    segmentation[IndicesToIndex(ix - 1, iy, iz)] = 2;
                }
                if (segmentation[IndicesToIndex(ix + 1, iy, iz)] == 1) {
                    NewSurfaceVoxel(IndicesToIndex(ix + 1, iy, iz), ix + 1, iy, iz);
                    segmentation[IndicesToIndex(ix + 1, iy, iz)] = 2;
                }
                if (segmentation[IndicesToIndex(ix, iy - 1, iz)] == 1) {
                    NewSurfaceVoxel(IndicesToIndex(ix, iy - 1, iz), ix, iy - 1, iz);
                    segmentation[IndicesToIndex(ix, iy - 1, iz)] = 2;
                }
                if (segmentation[IndicesToIndex(ix, iy + 1, iz)] == 1) {
                    NewSurfaceVoxel(IndicesToIndex(ix, iy + 1, iz), ix, iy + 1, iz);
                    segmentation[IndicesToIndex(ix, iy + 1, iz)] = 2;
                }
                if (segmentation[IndicesToIndex(ix, iy, iz - 1)] == 1) {
                    NewSurfaceVoxel(IndicesToIndex(ix, iy, iz - 1), ix, iy, iz - 1);
                    segmentation[IndicesToIndex(ix, iy, iz - 1)] = 2;
                }
                if (segmentation[IndicesToIndex(ix, iy, iz + 1)] == 1) {
                    NewSurfaceVoxel(IndicesToIndex(ix, iy, iz + 1), ix, iy, iz + 1);
                    segmentation[IndicesToIndex(ix, iy, iz + 1)] = 2;
                }

                // remove this from the surface voxels
                RemoveSurfaceVoxel(ptr);
                changed += 1;
            }
        }
        DestroyPointList(&deletable_points);
    }


    // return the number of changes
    return changed;
}



static void SequentialThinning(void)
{
    // create a vector of surface voxels
    CollectSurfaceVoxels();
    int iteration = 0;
    long changed = 0;
    do {
        changed = ThinningIterationStep();
        iteration++;
        if (print_verbose) printf("\n  thinning step: %3d.    (deleted point(s): %6ld)", iteration, changed);
    } while (changed);
}



static bool IsEndpoint(long ix, long iy, long iz)
{
    long nneighbors = 0;
    for (long iw = iz - 1; iw <= iz + 1; ++iw) {
        for (long iv = iy - 1; iv <= iy + 1; ++iv) {
            for (long iu = ix - 1; iu <= ix + 1; ++iu) {
                long linear_index = IndicesToIndex(iu, iv, iw);
                if (segmentation[linear_index]) {
                    nneighbors++;
                }
            }
        }
    }

    if (nneighbors == 2) return true;
    else return false;
}



void CppTopologicalThinning(const char *prefix, long resolution[3], const char *lookup_table_directory, bool benchmark)
{
    // initialize all of the lookup tables
    InitializeLookupTables(lookup_table_directory);

    // read the topologically downsampled file
    char input_filename[4096];
    if (benchmark) sprintf(input_filename, "topological/benchmarks/%s-topological-downsample-%ldx%ldx%ld.bytes", prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z]);
    else sprintf(input_filename, "topological/%s-topological-downsample-%ldx%ldx%ld.bytes", prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z]);

    // open the input file
    FILE *rfp = fopen(input_filename, "rb");
    if (!rfp) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }

    // read the size and number of segments
    if (fread(&zres, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
    if (fread(&yres, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
    if (fread(&xres, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }

    // add padding around each segment
    zres += 2;
    yres += 2;
    xres += 2;

    // set global indexing parameters
    nentries = zres * yres * xres;
    sheet_size = yres * xres;
    row_size = xres;

    // open the output filename
    char output_filename[4096];
    if (benchmark) sprintf(output_filename, "topological/benchmarks/%s-topological-downsample-%ldx%ldx%ld-thinning-skeleton.pts", prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z]);
    else sprintf(output_filename, "topological/%s-topological-downsample-%ldx%ldx%ld-thinning-skeleton.pts", prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z]);

    FILE *wfp = fopen(output_filename, "wb");
    if (!wfp) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }

    // go through all labels
    long max_label;
    if (fread(&max_label, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
    if (fwrite(&max_label, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }
    for (long label = 0; label < max_label; ++label) {
        segmentation = new unsigned char[nentries];
        for (long iv = 0; iv < nentries; ++iv) 
            segmentation[iv] = 0;

        long num;
        if (fread(&num, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }

        for (long iv = 0; iv < num; ++iv) {
            long element;
            if (fread(&element, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }

            // convert the element to non-cropped iz, iy, ix
            long iz = element / ((xres - 2) * (yres - 2));
            long iy = (element - iz * (xres - 2) * (yres - 2)) / (xres - 2);
            long ix = element % (xres - 2);

            // convert to cropped linear index
            element = (iz + 1) * sheet_size + (iy + 1) * row_size + ix + 1;

            segmentation[element] = 1;
        }

        if (print_verbose) printf("Number of points in original image for label %ld: %ld", label, num);

        // call the sequential thinning algorithm
        SequentialThinning();

        // count the number of remaining points
        num = 0;
        ListElement *LE = (ListElement *) surface_voxels.first;
        while (LE != NULL) {
            num++;
            LE = (ListElement *)LE->next;
        }
        if (print_verbose) printf("\nRemaining points for label %ld: %ld\n", label, num);

        // write the number of elements
        if (fwrite(&num, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }

        while (surface_voxels.first != NULL) {
            // get the surface voxels
            ListElement *LE = (ListElement *) surface_voxels.first;

            // get the coordinates for this skeleton point in the non-cropped segmentation
            long iz = LE->iz - 1;
            long iy = LE->iy - 1;
            long ix = LE->ix - 1;
            long iv = iz * (xres - 2) * (yres - 2) + iy * (xres - 2) + ix;

            if (fwrite(&iv, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }

            // remove this voxel
            RemoveSurfaceVoxel(LE);
        }


        // free memory
        delete[] segmentation;

        // reset global variables
        segmentation = NULL;
    }

    // close the files
    fclose(rfp);
    fclose(wfp);
    
    delete[] lut_simple;
    delete[] lut_isthmus;
}



static void ComputeDistanceFromBoundaryField(void)
{
    // allocate memory for bounday map and distance transform
    long *b = new long[nentries];
    for (long iz = 0; iz < zres; ++iz) {
        for (long iy = 0; iy < yres; ++iy) {
            for (long ix = 0; ix < xres; ++ix) {
                if (!segmentation[IndicesToIndex(ix, iy, iz)]) {
                    b[IndicesToIndex(ix, iy, iz)] = 0;
                    continue;
                }

                // inside voxels that are on the boundary have value 1 (based on TEASER paper figure 2)
                if ((ix == 0 or iy == 0 or iz == 0 or (ix == xres - 1) or (iy == yres - 1) or (iz == zres - 1)) ||
                    (ix > 0 and !segmentation[IndicesToIndex(ix - 1, iy, iz)]) ||
                    (iy > 0 and !segmentation[IndicesToIndex(ix, iy - 1, iz)]) ||
                    (iz > 0 and !segmentation[IndicesToIndex(ix, iy, iz - 1)]) ||
                    (ix < xres - 1 and !segmentation[IndicesToIndex(ix + 1, iy, iz)]) ||
                    (iy < yres - 1 and !segmentation[IndicesToIndex(ix, iy + 1, iz)]) ||
                    (iz < zres - 1 and !segmentation[IndicesToIndex(ix, iy, iz + 1)])) {
                    b[IndicesToIndex(ix, iy, iz)] = 1;
                }
                else {
                    b[IndicesToIndex(ix, iy, iz)] = infinity;
                }
            }
        }
    }

    // go along the z dimenion first for every (x, y) coordinate
    for (long ix = 0; ix < xres; ++ix) {
        for (long iy = 0; iy < yres; ++iy) {

            long k = 0;
            long *v = new long[zres + 1];
            double *z = new double[zres + 1];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < zres; ++q) {
                // label for jump statement
                zlabel:
                double s = ((b[IndicesToIndex(ix, iy, q)] + q * q) - (b[IndicesToIndex(ix, iy, v[k])] + v[k] * v[k])) / (float)(2 * q - 2 * v[k]);
                
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
            for (long q = 0; q < zres; ++q) {
                while (z[k + 1] < q)
                    ++k;

                DBF[IndicesToIndex(ix, iy, q)] = (q - v[k]) * (q - v[k]) + b[IndicesToIndex(ix, iy, v[k])];
            }

            // free memory 
            delete[] v;
            delete[] z;
        }
    }

    // update the boundary values with this distance
    for (long iz = 0; iz < zres; ++iz) {
        for (long iy = 0; iy < yres; ++iy) {
            for (long ix = 0; ix < xres; ++ix) {
                b[IndicesToIndex(ix, iy, iz)] = DBF[IndicesToIndex(ix, iy, iz)];
            }
        }
    }

    // go along the y dimension second for every (z, x) coordinate
    for (long iz = 0; iz < zres; ++iz) {
        for (long ix = 0; ix < xres; ++ix) {

            long k = 0;
            long *v = new long[yres + 1];
            double *z = new double[yres + 1];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < yres; ++q) {
                // label for jump statement
                ylabel:
                double s = ((b[IndicesToIndex(ix, q, iz)] + q * q) - (b[IndicesToIndex(ix, v[k], iz)] +  v[k] * v[k])) / (float)(2 * q - 2 * v[k]);
                
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
            for (long q = 0; q < yres; ++q) {
                while (z[k + 1] < q)
                    ++k;
            
                DBF[IndicesToIndex(ix, q, iz)] = (q - v[k]) * (q - v[k]) + b[IndicesToIndex(ix, v[k], iz)];
            }

            // free memory
            delete[] v;
            delete[] z;
        }
    }

    // update the boundary values with this distance
    for (long iz = 0; iz < zres; ++iz) {
        for (long iy = 0; iy < yres; ++iy) {
            for (long ix = 0; ix < xres; ++ix) {
                b[IndicesToIndex(ix, iy, iz)] = DBF[IndicesToIndex(ix, iy, iz)];
            }
        }
    }


    // go along the x dimension last for every (y, z) coordinate
    for (long iy = 0; iy < yres; ++iy) {
        for (long iz = 0; iz < zres; ++iz) {

            long k = 0;
            long *v = new long[xres + 1];
            double *z = new double[xres + 1];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < xres; ++q) {
                // label for jump statement
                xlabel:
                double s = ((b[IndicesToIndex(q, iy, iz)] + q * q) - (b[IndicesToIndex(v[k], iy, iz)] + v[k] * v[k])) / (float)(2 * q - 2 * v[k]);

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
            for (long q = 0;  q < xres; ++q) {
                while (z[k + 1] < q)
                    ++k;

                DBF[IndicesToIndex(q, iy, iz)] = (q - v[k]) * (q - v[k]) + b[IndicesToIndex(v[k], iy, iz)];
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
    long voxel_index = 0;
    while (!voxel_heap.IsEmpty()) {
        DijkstraData *current = voxel_heap.DeleteMin();
        voxel_index = current->iv;

        // visit all 26 neighbors of this index
        long ix, iy, iz;
        IndexToIndicies(voxel_index, ix, iy, iz);

        for (long iw = iz - 1; iw <= iz + 1; ++iw) {
            for (long iv = iy - 1; iv <= iy + 1; ++iv) {
                for (long iu = ix - 1; iu <= ix + 1; ++iu) {
                    // get the linear index for this voxel
                    long neighbor_index = IndicesToIndex(iu, iv, iw);

                    // skip if background
                    if (!segmentation[neighbor_index]) continue;
                    
                    // get the corresponding neighbor data
                    DijkstraData *neighbor_data = &(voxel_data[neighbor_index]);

                    // find the distance between these voxels
                    long deltaz = (iw - iz);
                    long deltay = (iv - iy);
                    long deltax = (iu - ix);

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

    // first call to this function needs to return the root and does not compute the skeleton
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
        if (print_verbose) printf("  Remaining inside voxels %ld...\n", inside_voxels);
        double farthest_pdrf = -1;
        long starting_voxel = -1;

        // find the farthest PDRF that is still inside
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
            IndexToIndicies(iv, ix, iy, iz);

            // get the skeleton path from this location to the root
            DijkstraData *current = &(voxel_data[starting_voxel]);

            while (!skeleton[current->iv]) {
                long ii, ij, ik;
                IndexToIndicies(current->iv, ii, ij, ik);
                // what is the distance between this skeleton location and the inside location
                double deltax = (ii - ix);
                double deltay = (ij - iy);
                double deltaz = (ik - iz);

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

        DijkstraData *current = &(voxel_data[starting_voxel]);
        while (!skeleton[current->iv]) {
            skeleton[current->iv] = 1;
            current = current->prev;
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
        if (DBF[iv] > M) M = DBF[iv]; 
    }

    // choose 5000 so that 3000 length voxel paths have correct floating point precision
    const double pdrf_scale = 5000;
    for (long iv = 0; iv < nentries; ++iv) {
        penalties[iv] = pdrf_scale * pow(1 - DBF[iv] / M, 16);
    }
}



void CppTeaserSkeletonization(const char *prefix, long resolution[3], bool benchmark)
{
        // read the topologically downsampled file
    char input_filename[4096];
    if (benchmark) sprintf(input_filename, "topological/benchmarks/%s-topological-downsample-%ldx%ldx%ld.bytes", prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z]);
    else sprintf(input_filename, "topological/%s-topological-downsample-%ldx%ldx%ld.bytes", prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z]);

    // open the input file
    FILE *rfp = fopen(input_filename, "rb");
    if (!rfp) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }

    // read the size and number of segments
    if (fread(&zres, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
    if (fread(&yres, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
    if (fread(&xres, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }

    // add padding around each segment
    zres += 2;
    yres += 2;
    xres += 2;

    // set global indexing parameters
    nentries = zres * yres * xres;
    sheet_size = yres * xres;
    row_size = xres;
    infinity = zres * zres + yres * yres + xres * xres;

    // open the output filename
    char output_filename[4096];
    if (benchmark) sprintf(output_filename, "topological/benchmarks/%s-topological-downsample-%ldx%ldx%ld-teaser-skeleton.pts", prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z]);
    else sprintf(output_filename, "topological/%s-topological-downsample-%ldx%ldx%ld-teaser-skeleton.pts", prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z]);

    FILE *wfp = fopen(output_filename, "wb");
    if (!wfp) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }

    // go through all labels
    long max_label;
    if (fread(&max_label, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
    if (fwrite(&max_label, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }
    for (long label = 0; label < max_label; ++label) {

        // find the number of elements in this segment
        long num;
        if (fread(&num, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
        if (!num) { fwrite(&num, sizeof(long), 1, wfp); continue; }

        // allocate memory for global variables
        segmentation = new unsigned char[nentries];
        skeleton = new unsigned char[nentries];
        penalties = new double[nentries];
        inside = new unsigned char[nentries];
        DBF = new double[nentries];
        for (long iv = 0; iv < nentries; ++iv) {
            segmentation[iv] = 0;
            skeleton[iv] = 0;
            penalties[iv] = 0;
            inside[iv] = 0;
            DBF[iv] = 0;
        }

        for (long iv = 0; iv < num; ++iv) {
            long element;
            if (fread(&element, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }


            // convert the element to non-cropped iz, iy, ix
            long iz = element / ((xres - 2) * (yres - 2));
            long iy = (element - iz * (xres - 2) * (yres - 2)) / (xres - 2);
            long ix = element % (xres - 2);

            // convert to cropped linear index
            element = (iz + 1) * sheet_size + (iy + 1) * row_size + ix + 1;

            segmentation[element] = 1;
            inside[element] = 1;
            inside_voxels++;
        }

        if (print_verbose) printf("Number of points in original image for label %ld: %ld\n", label, num);

        ComputeDistanceFromBoundaryField();

        // set any voxel as the source
        long source_voxel = -1;
        for (long iv = 0; iv < nentries; ++iv)
            if (inside[iv]) { source_voxel = iv; break; }

        // find a root voxel which is guaranteed to be at an extrema point
        long root_voxel = ComputeDistanceFromVoxelField(source_voxel);
        skeleton[root_voxel] = 1;
        inside[root_voxel] = 0;
        inside_voxels--;

        ComputePenalties();
        PDRF = new double[nentries];
        ComputeDistanceFromVoxelField(root_voxel);

        num = 0;
        for (long iv = 0; iv < nentries; ++iv) {
            if (skeleton[iv]) num++;
        }

        // write the number of elements
        if (fwrite(&num, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }

        for (long iv = 0; iv < nentries; ++iv) {
            if (!skeleton[iv]) continue;

            long ix, iy, iz;
            IndexToIndicies(iv, ix, iy, iz);
            --ix; --iy; --iz;

            long element = iz * (xres - 2) * (yres - 2) + iy * (xres - 2) + ix;
            if (fwrite(&element, sizeof(long), 1, wfp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }
        }

        // free memory
        delete[] segmentation;
        delete[] skeleton;
        delete[] penalties;
        delete[] PDRF;
        delete[] inside;
        delete[] DBF;

        // reset global variables
        segmentation = NULL;
        skeleton = NULL;
        penalties = NULL;
        PDRF = NULL;
        inside = NULL;
        DBF = NULL;
    }

    // close the files
    fclose(rfp);
    fclose(wfp);
}