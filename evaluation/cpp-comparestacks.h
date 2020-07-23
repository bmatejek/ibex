#pragma once
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <unordered_map>
#include <unordered_set>

struct EvaluationStats {
    std::unordered_map<long, std::unordered_map<long, long>> c;
    std::unordered_map<long, long> s;
    std::unordered_map<long, long> t;
    long nnonzero;
};


inline EvaluationStats *allocate_stats() {
    EvaluationStats* output = new EvaluationStats;
    output->c = std::unordered_map<long, std::unordered_map<long, long>>();
    output->s = std::unordered_map<long, long>();
    output->t = std::unordered_map<long, long>();
    output->nnonzero = 0;
    return output;
}

inline void deallocate_stats(EvaluationStats* pointer) {
    delete pointer;
}


inline static long NChoose2(long N)
{
    return N * (N - 1) / 2;
}

void CppGetStatsFromVolume(EvaluationStats *output, long *segmentation, long *gold, long grid_size[3], long *ground_truth_masks, long nmasks);

double* CppEvaluateStat(EvaluationStats *stat);

double *CppEvaluate(long *segmentation, long *gold, long grid_size[3], long *ground_truth_masks, long nmasks);