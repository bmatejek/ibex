import struct
import numpy as np

from ibex.evaluation.classification import *
from ibex.cnns.ebro.util import ReadGold, ReadCounters



# generate the baseline results by using Lee's old method
def Baseline(prefix_one, prefix_two, threshold, maximum_distance):
    # get the ground truth
    ground_truth = ReadGold(prefix_one, prefix_two, threshold, maximum_distance)
    ncandidates = len(ground_truth)

    # read the counters
    one_counts, two_counts, overlap = ReadCounters(prefix_one, prefix_two, threshold, maximum_distance)

    # iterate over all candidates
    labels = np.array(ground_truth)

    # write the May IARPA results to file
    with open('ebro/baseline/IARPA-MAY-{}-{}-{}-{}nm.probabilities'.format(prefix_one, prefix_two, threshold, maximum_distance), 'wb') as fd:
        ### predictions for IARPA 05/2017 ###
        predictions = np.zeros(ncandidates, dtype=np.uint8)
        fd.write(struct.pack('i', ncandidates))
        for iv in range(ncandidates):
            segment_one_score = float(overlap[iv]) / one_counts[iv]
            segment_two_score = float(overlap[iv]) / two_counts[iv]
            
            if segment_one_score > 0.8 or segment_two_score > 0.8: predictions[iv] = 1
            else: predictions[iv] = 0

            fd.write(struct.pack('d', predictions[iv]))

        print '\n#############################'
        print 'IARPA 05/2017 {} {}'.format(prefix_one.upper(), prefix_two.upper())
        print '#############################'
        PrecisionAndRecall(labels, predictions)

    with open('ebro/baseline/IARPA-AUG-{}-{}-{}-{}nm.probabilities'.format(prefix_one, prefix_two, threshold, maximum_distance), 'wb') as fd:
        ### predictions for IARPA 08/2017 ###
        predictions = np.zeros(ncandidates, dtype=np.uint8)
        fd.write(struct.pack('i', ncandidates))            
        for iv in range(ncandidates):
            segment_one_score = float(overlap[iv]) / one_counts[iv]
            segment_two_score = float(overlap[iv]) / two_counts[iv]

            if segment_one_score > 0.8 and segment_two_score > 0.8: predictions[iv] = 1
            else: predictions[iv] = 0

            fd.write(struct.pack('d', predictions[iv]))

        print '\n#############################'
        print 'IARPA 08/2017 {} {}'.format(prefix_one.upper(), prefix_two.upper())
        print '#############################'
        PrecisionAndRecall(labels, predictions)


def Ensemble(prefix_one, prefix_two, model_prefix, threshold, maximum_distance):
    # get the ground truth
    ground_truth = ReadGold(prefix_one, prefix_two, threshold, maximum_distance)
    ncandidates = len(ground_truth)
    labels = np.array(ground_truth)

    cnn_probabilities = np.zeros(ncandidates, dtype=np.float64)
    cnn_predictions = np.zeros(ncandidates, dtype=np.uint8)
    may_probabilities = np.zeros(ncandidates, dtype=np.float64)
    aug_probabilities = np.zeros(ncandidates, dtype=np.float64)
    
    with open('{}-{}-{}-{}-{}nm.probabilities'.format(model_prefix, prefix_one, prefix_two, threshold, maximum_distance), 'rb') as fd:
        ncandidates, = struct.unpack('i', fd.read(4))
        for iv in range(ncandidates):
            cnn_probabilities[iv], = struct.unpack('d', fd.read(8))
            if (cnn_probabilities[iv] > 0.5): cnn_predictions[iv] = 1
            else: cnn_predictions[iv] = 0
            
    with open('ebro/baseline/IARPA-MAY-{}-{}-{}-{}nm.probabilities'.format(prefix_one, prefix_two, threshold, maximum_distance), 'rb') as fd:
        ncandidates, = struct.unpack('i', fd.read(4))
        for iv in range(ncandidates):
            may_probabilities[iv], = struct.unpack('d', fd.read(8))

    with open('ebro/baseline/IARPA-AUG-{}-{}-{}-{}nm.probabilities'.format(prefix_one, prefix_two, threshold, maximum_distance), 'rb') as fd:
        ncandidates, = struct.unpack('i', fd.read(4))
        for iv in range(ncandidates):
            aug_probabilities[iv], = struct.unpack('d', fd.read(8))

    predictions = np.zeros(ncandidates, dtype=np.uint8)
    for iv in range(ncandidates):
        predictions[iv] = (cnn_probabilities[iv] > 0.5) and may_probabilities[iv]

        
    print '\n#############################'
    print 'CNN {} {}'.format(prefix_one.upper(), prefix_two.upper())
    print '#############################'    
    PrecisionAndRecall(labels, cnn_predictions)
    

    
    print '\n#############################'
    print 'ENSEMBLE {} {}'.format(prefix_one.upper(), prefix_two.upper())
    print '#############################'    
    PrecisionAndRecall(labels, predictions)
