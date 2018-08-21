void CppTopologicalThinning(const char *prefix, long resolution[3], const char *lookup_table_directory, bool benchmark);
void CppTeaserSkeletonization(const char *prefix, long resolution[3], bool benchmark);
void CppApplyUpsampleOperation(const char *prefix, long *input_segmentation, long resolution[3], const char *skeleton_algorithm, bool benchmark);