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
static const int IB_Z = 0;
static const int IB_Y = 1;
static const int IB_X = 2;
static const int IB_NDIMS = 3;



// make sure directories are set

static int set_directory = 0;
static char *output_directory = NULL;
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

static long zres;
static long yres;
static long xres;
static long nentries;
static long sheet_size;
static long row_size;



// lookup tables

static unsigned char *lut_simple;
static unsigned char *lut_isthmus;



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



void CppGenerateSkeletons(const char *prefix, long resolution[3], const char *lookup_table_directory)
{
    // make sure that the output directory is set
    if (!set_directory) { fprintf(stderr, "Failed to set the output dierctory!\n"); exit(-1); }

    // initialize all of the lookup tables
    InitializeLookupTables(lookup_table_directory);

    // read the topologically downsampled file
    char input_filename[4096];
    sprintf(input_filename, "topological/%s-topological-downsample-%ldx%ldx%ld.bytes", prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z]);

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
    sprintf(output_filename, "topological/%s-topological-downsample-%ldx%ldx%ld-thinning-skeleton.pts", prefix, resolution[IB_X], resolution[IB_Y], resolution[IB_Z]);

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

        printf("Number of points in original image for label %ld: %ld", label, num);

        // call the sequential thinning algorithm
        SequentialThinning();

        // count the number of remaining points
        num = 0;
        ListElement *LE = (ListElement *) surface_voxels.first;
        while (LE != NULL) {
            num++;
            LE = (ListElement *)LE->next;
        }
        printf("\nRemaining points for label %ld: %ld\n", label, num);

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
    }

    // close the files
    fclose(rfp);
    fclose(wfp);
    
    delete[] lut_simple;
    delete[] lut_isthmus;
}



void SetOutputDirectory(char *directory)
{
    output_directory = directory;
    set_directory = 1;
}
