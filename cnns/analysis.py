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
    if TP + FN + FP + TN == 0: print '    Accuracy: NaN'
    else: print '    Accuracy: {}'.format(float(TP + TN) / float(TP + FP + FN + TN))

    if TP + FP == 0 or TP + FN == 0:
        print '    F1 Score: NaN\n'
        return float('nan')
    
    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print '    F1 Score: {}\n'.format(f1_score)
    
    return f1_score
    
    
    
def CNNResultsSupplemental(problem):
    # get all of the trained networks for this problem that have some finished inference
    network_names = [model for model in sorted(glob.glob('architectures/{}*'.format(problem))) if len(glob.glob('{}/*inference.txt'.format(model)))]
    if not len(network_names):
        sys.stderr.write('No results found.\n')
        return

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

    optimal_training_f1_score = 0.0
    optimal_training_network = ''
    optimal_validation_f1_score = 0.0
    optimal_validation_network = ''
    optimal_testing_f1_score = 0.0
    optimal_testing_network = ''

    training_f1_scores = {}
    validation_f1_scores = {}
    testing_f1_scores = {}
    
    for network in network_names:
        if 'Kasthuri' in network or 'Fib25' in network: continue
        inference_filenames = sorted(glob.glob('{}/*inference.txt'.format(network)))
        
        training_results = []
        validation_results = []
        testing_results = []

        # parse each of the inference filenames
        for filename in inference_filenames:
            if not 'PNI' in filename: continue 
            
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
        training_f1_score = PrintResults(training_results, 'Training')
        validation_f1_score = PrintResults(validation_results, 'Validation')
        testing_f1_score = PrintResults(testing_results, 'Testing')

        if training_f1_score > optimal_training_f1_score:
            optimal_training_f1_score = training_f1_score
            optimal_training_network = network
        if validation_f1_score > optimal_validation_f1_score:
            optimal_validation_f1_score = validation_f1_score
            optimal_validation_network = network
        if testing_f1_score > optimal_testing_f1_score:
            optimal_testing_f1_score = testing_f1_score
            optimal_testing_network = network

        training_f1_scores[network] = training_f1_score
        validation_f1_scores[network] = validation_f1_score
        testing_f1_scores[network] = testing_f1_score
        
    print 'Optimal Training F1 Score: {}'.format(optimal_training_f1_score)
    print '  {}\n'.format(optimal_training_network)

    print 'Optimal Validation F1 Score: {}'.format(optimal_validation_f1_score)
    print '  {}\n'.format(optimal_validation_network)

    print 'Optimal Testing F1 Score: {}'.format(optimal_testing_f1_score)
    print '  {}\n'.format(optimal_testing_network)

    prev_diameter = ''

    print '\\begin{table*}'
    for network in network_names:
        if 'Kasthuri' in network or 'Fib25' in network: continue

        network_params = network.split('/')[1].split('-')

        diameter = 2 * int(network_params[1].strip('nm'))
        input_diameter = '{}nm'.format(diameter)
        width = '({})'.format(network_params[2].replace('x', ', '))


        if input_diameter != prev_diameter:
            if len(prev_diameter): print '\\end{tabular}'
            print '\\centering'
            print '\\caption{Results on \\SI{' + str(diameter) + '}{\\nano\\meter} diameter extracted regions of interest.}'
            print '\\begin{tabular}{c c c c} \\hline'
            print '\\textbf{Input Size} & \\textbf{Training F1 Score} & \\textbf{Validation F1 Score} & \\textbf{Testing F1 Score}  \\\\ \\hline'      
        print '%{}'.format(network)
        print '{} & {:0.4f} & {:0.4f} & {:0.4f} \\\\'.format(width, training_f1_scores[network], validation_f1_scores[network], testing_f1_scores[network])

        prev_diameter = input_diameter

    print '\\end{tabular}'
    print '\\end{table*}'
