import numpy as np
from numba import jit



def ThresholdPredictions(ground_truth, probabilities, output_filename):
    for ie in range(100):
        threshold = float(ie) / 100.0
        predictions = prob2pred(probabilities, threshold=threshold)

        TP, TN, FP, FN = PrecisionAndRecall(ground_truth, predictions)
        
        if TP + FP == 0:
            print '{}: {} - {} = NaN'.format(threshold, TP, FP)
        else:
            print '{}: {} - {} = {}'.format(threshold, TP, FP, float(TP) / float(TP + FP))



## TODO add compilation
#@jit(nopython=True)
def prob2pred(probabilities, threshold=0.5):
    # get the number of entries and the number of classes
    nentries = probabilities.shape[0]
    predictions = np.zeros(nentries, dtype=np.uint8)
    
    # iterate through every entry and every 
    for ie in range(nentries):
        if probabilities[ie] > threshold:
            predictions[ie] = 1
        else: 
            predictions[ie] = 0

    return predictions

#@jit(nopython=True)
def PrecisionAndRecall(ground_truth, predictions, output_filename=None):
    # make sure there are an equal number of elements
    assert (ground_truth.shape == predictions.shape)

    # make sure that the data is binary
    assert (np.amax(ground_truth) <= 1 and np.amax(predictions) <= 1)
    assert (np.amin(ground_truth) >= 0 and np.amax(predictions) >= 0)

    # set all of the counters to zero
    (TP, FP, FN, TN) = (0, 0, 0, 0)

    # iterate through every entry
    for ie in range(predictions.size):
        truth = ground_truth[ie]
        prediction = predictions[ie]

        if truth and prediction:
            TP += 1
        elif not truth and prediction:
            FP += 1
        elif truth and not prediction:
            FN += 1
        else:
            TN += 1
    
    ## TODO this is terrible
    if not output_filename == None:
        # open the filename to write
        with open(output_filename, 'a') as fd:
            fd.write('Positive Examples: {}\n'.format(TP + FN))
            fd.write('Negative Examples: {}\n'.format(FP + TN))
            fd.write('\n')
            fd.write('+--------------+----------------+\n')
            fd.write('|{:14s}|{:3s}{:13s}|\n'.format('', '', 'Prediction'))
            fd.write('+--------------+----------------+\n')
            fd.write('|{:14s}|  {:7s}{:7s}|\n'.format('', 'Merge', 'Split'))
            fd.write('|{:8s}{:5s} |{:7d}{:7d}  |\n'.format('', 'Merge', TP, FN))
            fd.write('| {:13s}|{:7s}{:7s}  |\n'.format('Truth', '', ''))
            fd.write('|{:8s}{:5s} |{:7d}{:7d}  |\n'.format('', 'Split', FP, TN))
            fd.write('+--------------+----------------+\n')

            if TP + FP == 0:
                fd.write('Precision: NaN\n')
            else:
                fd.write('Precision: {}\n'.format(float(TP) / float(TP + FP)))

            if TP + FN == 0:
                fd.write('Recall: NaN\n')
            else:
                fd.write('Recall: {}\n'.format(float(TP) / float(TP + FN)))

            fd.write('Accuracy: {}\n\n'.format(float(TP + TN) / float(TP + FP + FN + TN)))
        
    return TP, TN, FP, FN
