#include <stdio.h>
#include "cpp-seg2gold.h"
#include <map>


unsigned long *Seg2Gold(unsigned long *segmentation, unsigned int *gold, long nentries)
{
    // find the maximum segmentation value
    unsigned long max_segmentation_value = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (segmentation[iv] > max_segmentation_value)
            max_segmentation_value = segmentation[iv] + 1;
    }

    // create a mapping from segmentation to gold
    std::map<unsigned long, unsigned long> *mapping = new std::map<unsigned long, unsigned long>[max_segmentation_value];
    for (unsigned long iv = 0; iv < max_segmentation_value; ++iv)
        mapping[iv] = std::map<unsigned long, unsigned long>();

    // for every segment, gold label pair, increment the correct mapping
    for (long iv = 0; iv < nentries; ++iv) {
        mapping[segmentation[iv]][gold[iv]]++;
    }

    // find the largest gold label for each mapping
    unsigned long *segmentation_to_gold = new unsigned long[max_segmentation_value];
    for (unsigned long is = 0; is < max_segmentation_value; ++is) {
        unsigned long greatest_match = 0;
        unsigned long greatest_match_value = 0;
        for (std::map<unsigned long, unsigned long>::iterator it=mapping[is].begin(); it != mapping[is].end(); ++it) {
            // skip extracellular material
            if (it->first == 0) continue;

            // if this is the greatest match continue
            if (it->second > greatest_match_value) {
                greatest_match = it->first;
                greatest_match_value = it->second;
            }
        }
        segmentation_to_gold[is] = greatest_match;
    }

    // free memory
    delete[] mapping;

    return segmentation_to_gold;
}