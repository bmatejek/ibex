#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <set>
#include <time.h>
#include <math.h>
#include <set>
#include <assert.h>


// constant variables

static const int lookup_table_size = 1 << 23;
static const int NTHINNING_DIRECTIONS = 6;
static const int UP = 0;
static const int DOWN = 1;
static const int NORTH = 2;
static const int SOUTH = 3;
static const int EAST = 4;
static const int WEST = 5;
static const int NSMOOTHING_DIRECTIONS = 2;
static const int IB_Z = 0;
static const int IB_Y = 1;
static const int IB_X = 2;
static const int IB_NDIMS = 3;



// make sure that topological downsampling occurs

static int topological_downsampling = 0;
static int set_directory = 0;




static bool smoothing;
static bool first_only;
static int print_verbose = 1;



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



// global variables

static long high_res[3];
static long xres;
static long yres;
static long zres;
static long nentries;
static long sheet_size;
static long row_size;
static int ratio[3] = { -1, -1, -1 };
static char *output_directory = NULL;



// lookup tables

static unsigned char *lut_simple;
static unsigned char *lut_isthmus;
static unsigned char *lut_smoothing_r1;
static unsigned char *lut_smoothing_r2;





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



static std::set<long> *downsampled_voxels = NULL;
static unsigned char *segmentation = NULL;



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



static void InitializeLookupTables(char *lookup_table_directory)
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

    // read the smoothing r1 lookup table
    sprintf(lut_filename, "%s/lut_smoothing_r1.dat", lookup_table_directory);
    lut_smoothing_r1 = new unsigned char[lookup_table_size];
    lut_file = fopen(lut_filename, "rb");
    if (!lut_file) {
        fprintf(stderr, "Failed to read %s\n", lut_filename);
        exit(-1);
    }
    if (fread(lut_smoothing_r1, 1, lookup_table_size, lut_file) != lookup_table_size) {
        fprintf(stderr, "Failed to read %s\n", lut_filename);
        exit(-1);
    }
    fclose(lut_file);

    // read the smoothing r2 lookup table
    sprintf(lut_filename, "%s/lut_smoothing_r2.dat", lookup_table_directory);
    lut_smoothing_r2 = new unsigned char[lookup_table_size];
    lut_file = fopen(lut_filename, "rb");
    if (!lut_file) {
        fprintf(stderr, "Failed to read %s\n", lut_filename);
        exit(-1);
    }
    if (fread(lut_smoothing_r2, 1, lookup_table_size, lut_file) != lookup_table_size) {
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



static bool Smoothing26_6(unsigned int neighbors, unsigned int direction)
{
    if (direction == 0) return lut_smoothing_r1[(neighbors >> 3)] & char_mask[neighbors % 8];
    else return lut_smoothing_r2[(neighbors >> 3)] & char_mask[neighbors % 8];
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



static void DetectSmoothingBorderPoints(PointList *deletable_points, int direction)
{
    ListElement *LE = (ListElement *)surface_voxels.first;
    while (LE != NULL) {
        long iv = LE->iv;
        long ix = LE->ix;
        long iy = LE->iy;
        long iz = LE->iz;

        // not an isthmus
        unsigned int neighbors = Collect26Neighbors(ix, iy, iz);

        if (Smoothing26_6(neighbors, direction)) {
            Voxel voxel;
            voxel.iv = iv;
            voxel.ix = ix;
            voxel.iy = iy;
            voxel.iz = iz;
            AddToList(deletable_points, voxel, LE);
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



static long SmoothingIterationStep(void)
{
    long changed = 0;

    // iteratate through both directions
    for (int direction = 0; direction < NSMOOTHING_DIRECTIONS; ++direction) {
        PointList deletable_points;
        ListElement *ptr;

        CreatePointList(&deletable_points);
        DetectSmoothingBorderPoints(&deletable_points, direction);

        while (deletable_points.length) {
            Voxel voxel = GetFromList(&deletable_points, &ptr);

            long iv = voxel.iv;
            long ix = voxel.ix;
            long iy = voxel.iy;
            long iz = voxel.iz;

            unsigned int neighbors = Collect26Neighbors(ix, iy, iz);
            if (Smoothing26_6(neighbors, direction)) {
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

                // remove this from surface voxels
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
        if (smoothing) {
	  if (!first_only || !iteration) {
            int smoothing_iteration = 0;
            long smoothing_changed = 0;
            do {
                smoothing_changed = SmoothingIterationStep();
                smoothing_iteration++;
                if (print_verbose) printf("\n    smoothing step: %3d.    (deleted point(s): %6ld)", smoothing_iteration, smoothing_changed);
            } while (smoothing_changed);
	  }
        }
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



void CppGenerateSkeletons(long label, bool input_smoothing, bool input_first_only, char *lookup_table_directory)
{
    // make sure downsampling has occurred
    assert (topological_downsampling);
    assert (set_directory);

    smoothing = input_smoothing;
    first_only = input_first_only;
    
    // create the segmentation that will correspond to the skeletons
    segmentation = new unsigned char[nentries];
    for (long iv = 0; iv < nentries; ++iv)
        segmentation[iv] = 0;

    // iterate over the downsample set
    std::set<long>::iterator iter;
    long num = 0;
    for (iter = downsampled_voxels[label].begin(); iter != downsampled_voxels[label].end(); ++iter) {
        segmentation[*iter] = 1;
        num++;
    }
    if (print_verbose) printf("\n Number of object points in the original image: %ld\n", num);

    InitializeLookupTables(lookup_table_directory);

    clock_t t1, t2;
    t1 = clock();
    if (print_verbose) printf("\n Centerline extraction by sequential isthmus-based thinning ...");
    SequentialThinning();
    t2 = clock();
    if (print_verbose) printf("\n running time: %lf\n", ((float)t2 - (float)t1) / CLOCKS_PER_SEC);

    num = 0;
    ListElement *LE = (ListElement *) surface_voxels.first;
    while (LE != NULL) {
        num++;
        LE = (ListElement *)LE->next;
    }
    if (print_verbose) printf("\n\n Number of object points in the skeleton: %ld\n", num);

    // write the skeleton to an output folder
    char output_filename[4096];
    sprintf(output_filename, "%s/skeleton-%ld.pts", output_directory, label);

    FILE *fp = fopen(output_filename, "wb");
    if (!fp) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }

    // write the number of elements in the skeleton
    if (fwrite(&num, sizeof(long), 1, fp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }

    while (surface_voxels.first != NULL) {
        // get the surface voxel
        ListElement *LE = (ListElement *) surface_voxels.first;

        // get the coordinates for this skeleton point
        long ix, iy, iz;
        ix = LE->ix;
        iy = LE->iy;
        iz = LE->iz;

        // see if this location is an endpoint
        bool is_endpoint = IsEndpoint(ix, iy, iz);
        // subtract one to remove the padding
        --ix; --iy; --iz;

        // get the location in the high resolution grid
        ix *= ratio[IB_X];
        iy *= ratio[IB_Y];
        iz *= ratio[IB_Z];

        // negate the linear index for endpoints
        long iv = iz * (high_res[IB_X] * high_res[IB_Y]) + iy * (high_res[IB_X]) + ix;
        if (is_endpoint) iv = -1 * iv;

        if (fwrite(&iv, sizeof(long), 1, fp) != 1) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }

        // remove this surface voxel
        RemoveSurfaceVoxel(LE);
    }

    // close file
    fclose(fp);

    delete[] lut_simple;
    delete[] lut_isthmus;
    delete[] lut_smoothing_r1;
    delete[] lut_smoothing_r2;
    delete[] segmentation;
}



void CppTopologicalDownsampleData(long *input_segmentation, long input_high_res[3], int input_ratio[3])
{
    // make sure this function has not been called yet
    assert (not topological_downsampling);

    // get the number of entries at the high resolution
    long nhigh_entries = input_high_res[IB_Z] * input_high_res[IB_Y] * input_high_res[IB_X];

    // get the maximum segmentation in this segmentation
    long maximum_segmentation = 0;
    for (long iv = 0; iv < nhigh_entries; ++iv) {
        if (input_segmentation[iv] > maximum_segmentation) 
            maximum_segmentation = input_segmentation[iv] + 1;
    }

    // create a downsampled set for every segment
    downsampled_voxels = new std::set<long>[maximum_segmentation];
    if (!downsampled_voxels) exit(-1);
    for (long is = 0; is < maximum_segmentation; ++is)
        downsampled_voxels[is] = std::set<long>();

    // set the values for ratio and lower resolution values
    long low_res[IB_NDIMS];
    for (int dim = 0; dim < IB_NDIMS; ++dim) {
        high_res[dim] = input_high_res[dim];
        ratio[dim] = input_ratio[dim];
        // add two because there is padding in each direction
        low_res[dim] = ceil(high_res[dim] / ratio[dim]) + 2;
    }

    long iv = 0;
    for (long iz = 0; iz < high_res[IB_Z]; ++iz) {
        for (long iy = 0; iy < high_res[IB_Y]; ++iy) {
            for (long ix = 0; ix < high_res[IB_X]; ++ix, ++iv) {
                long label = input_segmentation[iv];
                // skip any extracellular material
                if (!label) continue;

                // find the low resolution location 
                // (add one because of padding on the low direction)
                long iw = iz / ratio[IB_Z] + 1;
                long iv = iy / ratio[IB_Y] + 1;
                long iu = ix / ratio[IB_X] + 1;

                // get the linear location
                long linear_index = iw * low_res[IB_X] * low_res[IB_Y] + iv * low_res[IB_X] + iu;

                downsampled_voxels[label].insert(linear_index);
            }
        }
    }

    // initialize convenient variables for thinning
    zres = low_res[IB_Z];
    yres = low_res[IB_Y];
    xres = low_res[IB_X];
    nentries = zres * yres * xres;
    sheet_size = yres * xres;
    row_size = xres;

    // make sure this function is called only once
    topological_downsampling = 1;
}



void SetDirectory(char *directory)
{
    assert (not set_directory);

    output_directory = directory;
    set_directory = 1;
}
