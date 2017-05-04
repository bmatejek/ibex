#include <stdio.h>
#include <stdlib.h>

unsigned long *CppMapLabels(unsigned long *segmentation, unsigned long *mapping, unsigned long nentries)
{
  unsigned long *updated_segmentation = new unsigned long[nentries];
  
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    updated_segmentation[iv] = mapping[segmentation[iv]];
  }

  return updated_segmentation;
}



unsigned long *CppRemoveSmallConnectedComponents(unsigned long *segmentation, int threshold, unsigned long nentries)
{
  if (threshold == 0) return segmentation;
  
  /* TODO can I assume that there are an integer number of voxels */

  // find the maximum label
  unsigned long max_segment_label = 0;
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    if (segmentation[iv] > max_segment_label) max_segment_label = segmentation[iv];
  }
  max_segment_label++;
  
  // create a counter array for the number of voxels
  int *nvoxels_per_segment = new int[max_segment_label];
  for (unsigned long iv = 0; iv < max_segment_label; ++iv) {
    nvoxels_per_segment[iv] = 0;
  }
  
  // count the number of voxels per segment
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    nvoxels_per_segment[segmentation[iv]]++;
  }
  
  // create the array for the updated segmentation
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    if (nvoxels_per_segment[segmentation[iv]] < threshold) segmentation[iv] = 0;
  }
  
  // free memory
  delete[] nvoxels_per_segment;
  
  return segmentation;
}
