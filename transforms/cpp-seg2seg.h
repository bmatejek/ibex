long *CppMapLabels(long *segmentation, long *mapping, unsigned long nentries);
long *CppRemoveSmallConnectedComponents(long *segmentation, int threshold, unsigned long nentries);
long *CppForceConnectivity(long *segmentation, long grid_size[3]);
void CppDownsampleMapping(const char *prefix, long *segmentation, long input_resolution[3], long output_resolution[3], long input_grid_size[3], bool benchmark);
