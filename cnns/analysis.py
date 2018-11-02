import glob
import os
import sys



def ParseInferenceFile(filename):
    with open(filename, 'r') as fd:
        lines = fd.readlines()

        # get the number of positive and negative examples
        npositives = int(lines[0].split(':')[1].strip())
        nnegatives = int(lines[1].split(':')[1].strip())

        # get TP, FN, FP, and TN
        true_merge = lines[7].split('|')[2]
        true_split = lines[9].split('|')[2]

        true_positives = int(true_merge.split()[0].strip())
        false_negatives = int(true_merge.split()[1].strip())

        false_positives = int(true_split.split()[0].strip())
        true_negatives = int(true_split.split()[1].strip())

        # get the precision, accuracy, and recall
        precision = float(lines[11].split(':')[1].strip())
        recall = float(lines[12].split(':')[1].strip())
        accuracy = float(lines[13].split(':')[1].strip())

        return true_positives, false_negatives, false_positives, true_negatives



def PrintResults(results, name):
    print '  {}'.format(name)

    # get the total precision and recall
    (TP, FN, FP, TN) = (0, 0, 0, 0)

    # go through all results and add
    for result in results:
        TP += result[0]
        FN += result[1]
        FP += result[2]
        TN += result[3]

    # format the output string
    print '    Positive Examples: {}'.format(TP + FN)
    print '    Negative Examples: {}\n'.format(FP + TN)
    print '    +--------------+----------------+'
    print '    |{:14s}|{:3s}{:13s}|'.format('', '', 'Prediction')
    print '    +--------------+----------------+'
    print '    |{:14s}|  {:7s}{:7s}|'.format('', 'Merge', 'Split')
    print '    |{:8s}{:5s} |{:7d}{:7d}  |'.format('', 'Merge', TP, FN)
    print '    | {:13s}|{:7s}{:7s}  |'.format('Truth', '', '')
    print '    |{:8s}{:5s} |{:7d}{:7d}  |'.format('', 'Split', FP, TN)
    print '    +--------------+----------------+'
    if TP + FP == 0: print '    Precision: NaN'
    else: print '    Precision: {}'.format(float(TP) / float(TP + FP))
    if TP + FN == 0: print '    Recall: NaN'
    else: print '    Recall: {}'.format(float(TP) / float(TP + FN))
    if TP + FN + FP + TN == 0: print '    Accuracy: NaN\n'
    else: print '    Accuracy: {}\n'.format(float(TP + TN) / float(TP + FP + FN + TN))



def CNNResults(problem):
    # get all of the trained networks for this problem
    network_names = sorted(glob.glob('architectures/{}*'.format(problem)))

    # hardcoded for PNI data, create new method to store this
    subsets = {
        'train-one': 'training',
        'train-two': 'training',
        'train-three': 'training',
        'train-four': 'training',
        'train-five': 'validation',
        'train-six': 'validation',
        'validation-one': 'validation',
        'test-one': 'testing',
        'test-two': 'testing'
    }

    for network in network_names:
        inference_filenames = sorted(glob.glob('{}/*inference.txt'.format(network)))
        
        training_results = []
        validation_results = []
        testing_results = []

        # parse each of the inference filenames
        for filename in inference_filenames:
            prefix = '-'.join(filename.split('/')[-1].split('-')[1:-1])

            # get the dataset from the prefix name
            dataset = '-'.join(prefix.split('-')[1:3])
            
            # get the subset this belongs to 
            subset = subsets[dataset]

            # parse the inference file
            TP, FN, FP, TN = ParseInferenceFile(filename)

            if subset == 'training':
                training_results.append((TP, FN, FP, TN))
            elif subset == 'validation':
                validation_results.append((TP, FN, FP, TN))
            elif subset == 'testing':
                testing_results.append((TP, FN, FP, TN))
            else:
                sys.stderr.write('Unrecognized subset {}'.format(subset))
                sys.exit()

        # agglomerate all results
        print '{}'.format(network.split('/')[1])
        PrintResults(training_results, 'Training')
        PrintResults(validation_results, 'Validation')
        PrintResults(testing_results, 'Teseting')
