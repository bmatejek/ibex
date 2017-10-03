unsigned long *CppMapLabels(unsigned long *segmentation, unsigned long *mapping, unsigned long nentries);
unsigned long *CppRemoveSmallConnectedComponents(unsigned long *segmentation, int threshold, unsigned long nentries);
unsigned long *CppForceConnectivity(unsigned long *segmentation, long zres, long yres, long xres);