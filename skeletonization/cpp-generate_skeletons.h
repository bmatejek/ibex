#ifndef __CPP_GENERATE_SKELETONS__
#define __CPP_GENERATE_SKELETONS__

#include <ctime>

// function calls across cpp files
void CppTopologicalThinning(const char *prefix, long skeleton_resolution[3], const char *lookup_table_directory, bool benchmark);
void CppTeaserSetScale(double input_scale);
void CppTeaserSetBuffer(long input_buffer);
void CppTeaserSkeletonization(const char *prefix, long skeleton_resolution[3], bool benchmark);
void CppAStarSetMaxExpansion(double input_max_distance);
void CppApplyUpsampleOperation(const char *prefix, long *input_segmentation, long skeleton_resolution[3], long output_resolution[3], const char *skeleton_algorithm, bool benchmark);
void CppNaiveUpsampleOperation(const char *prefix, long skeleton_resolution[3], const char *skeleton_algorithm, bool benchmark, double scale, long buffer);



// universal variables and functions

static const int IB_Z = 0;
static const int IB_Y = 1;
static const int IB_X = 2;

#endif