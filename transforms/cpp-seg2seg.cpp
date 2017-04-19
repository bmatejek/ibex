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
