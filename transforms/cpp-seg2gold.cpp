#include <stdio.h>
#include "cpp-seg2gold.h"
#include <map>


unsigned long *CppMapping(unsigned long *segmentation, unsigned int *gold, long nentries)
{
    // find the maximum segmentation value
    unsigned long max_segmentation_value = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (segmentation[iv] > max_segmentation_value)
            max_segmentation_value = segmentation[iv];
    }
    max_segmentation_value++;

    // find the maximum gold value
    unsigned long max_gold_value = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (gold[iv] > max_gold_value) 
            max_gold_value = gold[iv];
    }
    max_gold_value++;

    /* TODO way too memory expensive */
    unsigned long **seg2gold_overlap = new unsigned long *[max_segmentation_value];
    for (unsigned long is = 0; is < max_segmentation_value; ++is) {
        seg2gold_overlap[is] = new unsigned long[max_gold_value];
        for (unsigned long ig = 0; ig < max_gold_value; ++ig) {
            seg2gold_overlap[is][ig] = 0;
        }
    }

    // iterate over every voxel
    for (long iv = 0; iv < nentries; ++iv) {
        seg2gold_overlap[segmentation[iv]][gold[iv]]++;
    }

    // create the mapping
    unsigned long *segmentation_to_gold = new unsigned long[max_segmentation_value];
    for (unsigned long is = 0; is < max_segmentation_value; ++is) {
        unsigned long gold_id = 0;
        unsigned long gold_max_value = 0;
        // do not consider extra cellular locations
        // if the entire thing is extracellular gold_id will be one
        for (unsigned long ig = 1; ig < max_gold_value; ++ig) {
            if (seg2gold_overlap[is][ig] > gold_max_value) {
                gold_max_value = seg2gold_overlap[is][ig];
                gold_id = ig;
            }
        }

        segmentation_to_gold[is] = gold_id;
    }

    // free memory
    for (unsigned long is = 0; is < max_segmentation_value; ++is) {
        delete[] seg2gold_overlap[is];
    }
    delete[] seg2gold_overlap;

    return segmentation_to_gold;
}