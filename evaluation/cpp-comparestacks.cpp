#include <ctime>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <map>




// constant variables

static const int IB_Z = 0;
static const int IB_Y = 1;
static const int IB_X = 2;




static long NChoose2(long N) 
{
    return N * (N - 1) / 2;
}



void CppEvaluate(long *segmentation, long *gold, long resolution[3], bool mask_ground_truth)
{
    // get convenient variables
    long zres = resolution[IB_Z];
    long yres = resolution[IB_Y];
    long xres = resolution[IB_X];
    long nentries = zres * yres * xres;
    long nnonzero = nentries;


    // start stopwatch
    clock_t t1, t2;
    t1 = clock();

    // update the number of nonzero if mask is on 
    for (long iv = 0; iv < nentries; ++iv) {
        if (!gold[iv]) nnonzero--;
    }

    // get the maximum value for the segmentation and gold volumes
    long max_segment = 0;
    long max_gold = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (segmentation[iv] > max_segment) max_segment = segmentation[iv];
        if (gold[iv] > max_gold) max_gold = gold[iv];
    }
    ++max_segment;
    ++max_gold;

    // get an array for segment values that occur
    bool *segment_exists = new bool[max_segment];
    bool *gold_exists = new bool[max_gold];
    for (long is = 0; is < max_segment; ++is)
        segment_exists[is] = false;
    for (long ig = 0; ig < max_gold; ++ig)
        gold_exists[ig] = false;
    for (long iv = 0; iv < nentries; ++iv) {
        segment_exists[segmentation[iv]] = true;
        gold_exists[gold[iv]] = true;
    }
    // populate values for joint sets
    std::map<long, std::map<long, long> > c = std::map<long, std::map<long, long> >();   
    for (long is = 0; is < max_segment; ++is) {
        if (segment_exists[is]) {
            c.insert(std::pair<long, std::map<long, long> >(is, std::map<long, long>()));
        }
    }
    for (long iv = 0; iv < nentries; ++iv)
        c[segmentation[iv]][gold[iv]]++;
    std::map<long, long> s = std::map<long, long>();
    std::map<long, long> t = std::map<long, long>();
    for (long is = 0; is < max_segment; ++is) {
        if (not segment_exists[is]) continue;
        for (long ig = mask_ground_truth; ig < max_gold; ++ig) {
            if (not gold_exists[ig]) continue;
            if (not c[is].count(ig)) continue;
            s[is] += c[is][ig];
        }
    }
    for (long ig = mask_ground_truth; ig < max_gold; ++ig) {
        if (not gold_exists[ig]) continue;
        for (long is = 0; is  < max_segment; ++is) {
            if (not segment_exists[is]) continue;
            if (not c[is].count(ig)) continue;
            t[ig] += c[is][ig];
        }
    }

    /*long **c = new long *[max_segment];
    for (long is = 0; is < max_segment; ++is) {
        c[is] = new long[max_gold];
        for (long ig = 0; ig < max_gold; ++ig) {
            c[is][ig] = 0;
        }
    }
    for (long iv = 0; iv < nentries; ++iv) {
        c[segmentation[iv]][gold[iv]]++;
    }

    // populate values for s and t sets
    long *s = new long[max_segment];
    for (long is = 0; is < max_segment; ++is) {
        s[is] = 0;
        for (long ig = mask_ground_truth; ig < max_gold; ++ig) {
            s[is] += c[is][ig];
        }
    }
    long *t = new long[max_gold];
    for (long ig = mask_ground_truth; ig < max_gold; ++ig) {
        t[ig] = 0;
        for (long is = 0; is < max_segment; ++is) {
            t[ig] += c[is][ig];
        }
    }*/


    // calculate the number of true and false positives and negatives
    long TP = 0;
    for (long is = 0; is < max_segment; ++is) {
        if (not segment_exists[is]) continue;
        for (long ig = mask_ground_truth; ig < max_gold; ++ig) {
            if (not gold_exists[is]) continue;
            if (not c[is].count(ig)) continue;
            TP += NChoose2(c[is][ig]);
        }
    }
    long TP_FP = 0;
    for (long is = 0; is < max_segment; ++is) {
        if (not segment_exists[is]) continue;
        TP_FP += NChoose2(s[is]);
    }
    long TP_FN = 0;
    for (long ig = mask_ground_truth; ig < max_gold; ++ig) {
        if (not gold_exists[ig]) continue;
        TP_FN += NChoose2(t[ig]);
    }
    long FP = TP_FP - TP;
    long FN = TP_FN - TP;

    printf("Rand Error Full: %lf\n", (FP + FN) / (double) (NChoose2(nnonzero)));
    printf("Rand Error Merge: %lf\n", FP / (double) (NChoose2(nnonzero)));
    printf("Rand Error Split: %lf\n", FN / (double) (NChoose2(nnonzero)));



    // calculate the variation of information
    double VI_split = 0.0;
    double VI_merge = 0.0;

    for (long is = 0; is < max_segment; ++is) {
        if (not segment_exists[is]) continue;
        double spi = s[is] / (double)nnonzero;
        for (long ig = mask_ground_truth; ig < max_gold; ++ig) {
            if (not gold_exists[is]) continue;
            if (not c[is].count(ig)) continue;
            double tpj = t[ig] / (double)nnonzero;
            double pij = c[is][ig] / (double)nnonzero;

            VI_split = VI_split - pij * log(pij / tpj);
            VI_merge = VI_merge - pij * log(pij / spi);
        }
    }

    printf("Variation of Information Full: %lf\n", VI_split + VI_merge);
    printf("Variation of Information Merge: %lf\n", VI_merge);
    printf("Variation of Information Split: %lf\n", VI_split);

    // free memory
    delete[] segment_exists;
    delete[] gold_exists;

    // end stopwatch
    t2 = clock();
    printf("\n running time: %lf\n", ((float)t2 - (float)t1) / CLOCKS_PER_SEC);
}