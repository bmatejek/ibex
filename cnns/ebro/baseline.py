from ibex.evaluation.classification import *
from ibex.cnns.ebro.util import ReadGold, ReadCounters



# generate the baseline results by using Lee's old method
def Baseline(prefix_one, prefix_two, threshold, maximum_distance):
    print '---------------------------------------------------'
    print '                 {} {}                 '.format(prefix_one, prefix_two)
    print '---------------------------------------------------'

    # get the ground truth
    ground_truth = ReadGold(prefix_one, prefix_two, threshold, maximum_distance)
    ncandidates = len(ground_truth)

    # read the counters
    one_counts, two_counts, overlap = ReadCounters(prefix_one, prefix_two, threshold, maximum_distance)

    # iterate over all candidates
    labels = np.array(ground_truth)

    ### predictions for IARPA 05/2017 ###
    predictions = np.zeros(ncandidates, dtype=np.uint8)
    for iv in range(ncandidates):
        segment_one_score = float(overlap[iv]) / one_counts[iv]
        segment_two_score = float(overlap[iv]) / two_counts[iv]

        if segment_one_score > 0.8 or segment_two_score > 0.8:
            predictions[iv] = 1
        else:
            predictions[iv] = 0

    print 'IARPA 05/2017'
    PrecisionAndRecall(labels, predictions)


    ### predictions for IARPA 08/2017 ###
    predictions = np.zeros(ncandidates, dtype=np.uint8)
    for iv in range(ncandidates):
        segment_one_score = float(overlap[iv]) / one_counts[iv]
        segment_two_score = float(overlap[iv]) / two_counts[iv]

        if segment_one_score > 0.8 and segment_two_score > 0.8:
            predictions[iv] = 1
        else:
            predictions[iv] = 0

    print 'IARPA 08/2017'
    PrecisionAndRecall(labels, predictions)
