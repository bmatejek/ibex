void CppMapLabels(long *segmentation, long *mapping, unsigned long nentries);
long *CppRemoveSmallConnectedComponents(long *segmentation, int threshold, unsigned long nentries);
long *CppForceConnectivity(long *segmentation, long zres, long yres, long xres);
void CppTopologicalDownsample(const char *prefix, long *segmentation, long input_resolution[3], long output_resolution[3], long input_zres, long input_yres, long input_xres);
void CppTopologicalUpsample(const char *prefix, long *segmentation, long input_resolution[3], long output_resolution[3], long input_zres, long input_yres, long input_xres);
