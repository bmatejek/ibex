#include <stdio.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#define IB_X 2
#define IB_Y 1
#define IB_Z 0


static long nentries;
static long sheet_size;
static long row_size;



static double threshold = 0.2;


static void IndexToIndicies(long iv, long &ix, long &iy, long &iz)
{
    iz = iv / sheet_size;
    iy = (iv - iz * sheet_size) / row_size;
    ix = iv % row_size;
}



static long IndiciesToIndex(long ix, long iy, long iz)
{
    return iz * sheet_size + iy * row_size + ix;
}



static long offsets[26];



static void PopulateOffsets(void)
{
    offsets[0] = -1 * sheet_size - row_size - 1;
    offsets[1] = -1 * sheet_size - row_size;
    offsets[2] = -1 * sheet_size - row_size + 1;
    offsets[3] = -1 * sheet_size - 1;
    offsets[4] = -1 * sheet_size;
    offsets[5] = -1 * sheet_size + 1;
    offsets[6] = -1 * sheet_size + row_size - 1;
    offsets[7] = -1 * sheet_size + row_size;
    offsets[8] = -1 * sheet_size + row_size + 1;

    offsets[9] = -1 * row_size - 1;
    offsets[10] = -1 * row_size;
    offsets[11] = -1 * row_size + 1;
    offsets[12] = - 1;
    offsets[13] = + 1;
    offsets[14] = row_size - 1;
    offsets[15] = row_size;
    offsets[16] = row_size + 1;

    offsets[17] = sheet_size - row_size - 1;
    offsets[18] = sheet_size - row_size;
    offsets[19] = sheet_size - row_size + 1;
    offsets[20] = sheet_size - 1;
    offsets[21] = sheet_size;
    offsets[22] = sheet_size + 1;
    offsets[23] = sheet_size + row_size - 1;
    offsets[24] = sheet_size + row_size;
    offsets[25] = sheet_size + row_size + 1;
}



long *CppLocateSingletons(long *segmentation, long grid_size[3])
{
    // set useful global variables
    nentries = grid_size[IB_Z] * grid_size[IB_Y] * grid_size[IB_X];
    sheet_size = grid_size[IB_Y] * grid_size[IB_X];
    row_size = grid_size[IB_X];

    // find the total number of labels
    long maximum_label = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (segmentation[iv] > maximum_label) maximum_label = segmentation[iv];
    }
    ++maximum_label;

    // every label will start with long -2, as soon as it appears in a z slice it will be updated with that value
    // as soon as another z slice appears for this segment (i.e., singleton[segmentation[iv]] != -1 or iz), it is not a singleton (label -1)
    long *singleton = new long[maximum_label];
    for (long is = 0; is < maximum_label; ++is)
        singleton[is] = -2;

    long iv = 0;
    for (long iz = 0; iz < grid_size[IB_Z]; ++iz) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
            for (long ix = 0; ix < grid_size[IB_X]; ++ix, ++iv) {
                long label = segmentation[iv];
                if (singleton[label] == -2) singleton[label] = iz;
                else if (singleton[label] != iz) singleton[label] = -1;
            }
        }
    }

    long nsingletons = 0;
    for (long is = 0; is < maximum_label; ++is) {
        if (singleton[is] >= 0) nsingletons++;
    }

    printf("No. singletons: %ld\n", nsingletons);









/*
    std::unordered_map<long, long> voxel_matches = std::unordered_map<long, long>();

    for (long iz = 0; iz < grid_size[IB_Z] - 1; ++iz) {
        for (long iy = 0; iy < grid_size[IB_Y]; ++iy) {
            for (long ix = 0; ix <grid_size[IB_X]; ++ix) {
                long segment = segmentation[IndiciesToIndex(ix, iy, iz)];
                long neighbor_segment = segmentation[IndiciesToIndex(ix, iy, iz + 1)];

                // skip over non singleton elements
                if (not is_singleton[segment] or not is_singleton[neighbor_segment]) continue;

                if (neighbor_segment < segment) {
                    long tmp = neighbor_segment;
                    neighbor_segment = segment;
                    segment = tmp;
                }

                voxel_matches[segment * maximum_label + neighbor_segment]++;
            }
        }
    }

    std::vector<long> pairs = std::vector<long>();

    std::unordered_map<long, long>::iterator it;
    for (it = voxel_matches.begin(); it != voxel_matches.end(); ++it) {
        long indicies = it->first;
        long first_index = indicies / maximum_label;
        long second_index = indicies % maximum_label;

        long nvoxel_matches = it->second;

        long total_voxels = nvoxels_per_segment[first_index] + nvoxels_per_segment[second_index] - nvoxel_matches;

        double overlap = nvoxel_matches / (double)(total_voxels);

        if (overlap > threshold) {
            pairs.push_back(first_index);
            pairs.push_back(second_index);
        }
    }

    long *matches = new long[pairs.size() + 1];
    matches[0] = pairs.size();
    for (unsigned long iv = 0; iv < pairs.size(); ++iv) {
        matches[iv + 1] = pairs[iv];
    }

    delete[] nvoxels_per_segment;
*/
    return NULL;
}