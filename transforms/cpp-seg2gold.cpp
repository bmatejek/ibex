#include <stdio.h>
#include "cpp-seg2gold.h"
#include <map>


long *CppMapping(long *segmentation, int *gold, long nentries, double match_threshold, double nonzero_threshold)
{
    // find the maximum segmentation value
    long max_segmentation_value = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (segmentation[iv] > max_segmentation_value)
            max_segmentation_value = segmentation[iv];
    }
    max_segmentation_value++;

    // find the maximum gold value
    long max_gold_value = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (gold[iv] > max_gold_value) 
            max_gold_value = gold[iv];
    }
    max_gold_value++;

    // find the number of voxels per segment
    unsigned long *nvoxels_per_segment = new unsigned long[max_segmentation_value];
    for (long iv = 0; iv < max_segmentation_value; ++iv)
        nvoxels_per_segment[iv] = 0;
    for (long iv = 0; iv < nentries; ++iv)
      nvoxels_per_segment[segmentation[iv]]++;

    std::map<long, std::map<long, long> > seg2gold_overlap = std::map<long, std::map<long, long> >();	
    for (long is = 0; is < max_segmentation_value; ++is) {
      if (nvoxels_per_segment[is]) { seg2gold_overlap.insert(std::pair<long, std::map<long, long> >(is, std::map<long, long>())); }
    }
    
    for (long iv = 0; iv < nentries; ++iv) {
      seg2gold_overlap[segmentation[iv]][gold[iv]]++;
    }
    


    /* TODO way too memory expensive */
    long **seg2gold_overlap_old = new long *[max_segmentation_value];
    for (long is = 0; is < max_segmentation_value; ++is) {
        seg2gold_overlap_old[is] = new long[max_gold_value];
        for (long ig = 0; ig < max_gold_value; ++ig) {
            seg2gold_overlap_old[is][ig] = 0;
        }
    }

    // iterate over every voxel
    for (long iv = 0; iv < nentries; ++iv) {
      seg2gold_overlap_old[segmentation[iv]][gold[iv]]++;
    }

    for (long is = 0; is < max_segmentation_value; ++is) {
      for (long ig = 0; ig < max_gold_value; ++ig) {
	if (seg2gold_overlap_old[is][ig]) 
	  if (seg2gold_overlap[is][ig] != seg2gold_overlap_old[is][ig]) printf("NO!\n");
      }
    }
	  

    
    // create the mapping
    long *segmentation_to_gold = new long[max_segmentation_value];
    for (long is = 0; is < max_segmentation_value; ++is) {
        if (!nvoxels_per_segment[is]) { segmentation_to_gold[is] = 0; continue; }
	long gold_id = 0;
	long gold_max_value = 0;

        // only gets label of 0 if the number of non zero voxels is below threshold
	for (std::map<long, long>::iterator iter = seg2gold_overlap[is].begin(); iter != seg2gold_overlap[is].end(); ++iter) {
	  if (not iter->first) continue;
	  if (iter->second > gold_max_value) {
	    gold_max_value = iter->second;
	    gold_id = iter->first;
	  }
	}

        // the number of matching gold values divided by the number of non zero pixels must be greater than the match threshold
        if (gold_max_value / (double)(nvoxels_per_segment[is] - seg2gold_overlap[is][0]) < match_threshold) segmentation_to_gold[is] = 0;
        // number of non zero pixels must be greater than the nonzero threshold
        else if ((double)(nvoxels_per_segment[is] - seg2gold_overlap[is][0]) / nvoxels_per_segment[is] < nonzero_threshold) segmentation_to_gold[is] = 0;
        else segmentation_to_gold[is] = gold_id;
    }

    // free memory
    delete[] nvoxels_per_segment;
    
    return segmentation_to_gold;
}
