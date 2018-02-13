void SetDirectory(char *directory);
void CppTopologicalDownsampleData(long *input_segmentation, long high_res[3], int ratio[3]);
void CppGenerateSkeletons(long label, char *lookup_table_directory);