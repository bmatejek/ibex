import numpy as np
from numba import jit



@jit(nopython=True)
def Prob2Pred(probabilities, threshold=0.5):
    nentries = probabilities.shape[0]
    predictions = np.zeros(nentries, dtype=np.uint8)

    for ie in range(nentries):
        if probabilities[ie] > threshold: predictions[ie] = True
        else: predictions[ie] = False

    return predictions


def PrecisionAndRecallCurve(ground_truth, probabilities):
    results = sorted(zip(probabilities, ground_truth), reverse=True)

    precisions = []
    recalls = []

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for (probability, label) in results:
        if probability > 0.5 and label: TP += 1
        if probability > 0.5 and not label: FP += 1
        if probability < 0.5 and label: FN += 1
        if probability < 0.5 and not label: TN += 1

        if TP + FP == 0 or TP + FN == 0: continue
        precisions.append(TP / float(TP + FP))
        recalls.append(TP / float(TP + FN))

    return precisions, recalls
        
    
    

def ReceiverOperatingCharacteristicCurve(ground_truth, probabilities):
    results = sorted(zip(probabilities, ground_truth), reverse=True)

    true_positive_rates = []
    false_positive_rates = []

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    for (probability, label) in results:
        if probability > 0.5 and label: TP += 1
        if probability > 0.5 and not label: FP += 1
        if probability < 0.5 and label: FN += 1
        if probability < 0.5 and not label: TN += 1

        if TP + FN == 0 or TN + FP == 0: continue
        true_positive_rates.append(TP / float(TP + FN))
        false_positive_rates.append(1.0 - TN / float(TN + FP))

    true_positive_rates.append(0)
    false_positive_rates.append(0)
        
    return true_positive_rates, false_positive_rates
    
    
    
def PrecisionAndRecall(ground_truth, predictions, output_filename=None, binary=True):
    assert (ground_truth.shape == predictions.shape)

    # set all of the counters to zero
    (TP, FP, FN, TN) = (0, 0, 0, 0)

    # iterate through every entry
    for ie in range(predictions.size):
        # get the label and the prediction
        label = ground_truth[ie]
        prediction = predictions[ie]

        # some slots are used as throwaways
        if binary and not (label == 0 or label == 1): continue

        # increment the proper variables
        if label and prediction: TP += 1
        elif not label and prediction: FP += 1
        elif label and not prediction: FN += 1
        else: TN += 1
    
    # format the output string
    output_string = 'Positive Examples: {}\n'.format(TP + FN)
    output_string += 'Negative Examples: {}\n\n'.format(FP + TN)
    output_string += '+--------------+----------------+\n'
    output_string += '|{:14s}|{:3s}{:13s}|\n'.format('', '', 'Prediction')
    output_string += '+--------------+----------------+\n'
    output_string += '|{:14s}|  {:7s}{:7s}|\n'.format('', 'Merge', 'Split')
    output_string += '|{:8s}{:5s} |{:7d}{:7d}  |\n'.format('', 'Merge', TP, FN)
    output_string += '| {:13s}|{:7s}{:7s}  |\n'.format('Truth', '', '')
    output_string += '|{:8s}{:5s} |{:7d}{:7d}  |\n'.format('', 'Split', FP, TN)
    output_string += '+--------------+----------------+\n'
    if TP + FP == 0: output_string += 'Precision: NaN\n'
    else: output_string += 'Precision: {}\n'.format(float(TP) / float(TP + FP))
    if TP + FN == 0: output_string += 'Recall: NaN\n'
    else: output_string += 'Recall: {}\n'.format(float(TP) / float(TP + FN))
    output_string += 'Accuracy: {}'.format(float(TP + TN) / float(TP + FP + FN + TN))

    # output the string to the output file and standard out
    print output_string
    if not output_filename == None:
        with open(output_filename, 'w') as fd:
            fd.write(output_string)
