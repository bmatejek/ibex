import glob
import os

class NetworkResult:
    def __init__(self, name, architecture, parameters, results):
        self.name = name
        self.architecture = architecture
        self.parameters = parameters
        self.results = results

    def Precision(self):
        return [result.precision for result in self.results]

    def Recall(self):
        return [result.recall for result in self.results]

    def Accuracy(self):
        return [result.accuracy for result in self.results]

    def __cmp__(self, other):
        self_width = self.architecture.input_size.strip('()').split(', ')
        other_width = other.architecture.input_size.strip('()').split(', ')


        self_input_size = int(self_width[0]) * int(self_width[1]) * int(self_width[2]) * int(self_width[3])
        other_input_size = int(other_width[0]) * int(other_width[1]) * int(other_width[2]) * int(other_width[3])

        if self.name > other.name: return 1
        else: return -1
        
        if self_input_size > other_input_size: return 1
        elif self_input_size < other_input_size: return -1
        elif self.parameters['iterations'] > other.parameters['iterations']: return 1
        elif self.parameters['iterations'] < other.parameters['iterations']: return -1
        elif self.parameters['initial_learning_rate'] < other.parameters['initial_learning_rate']: return 1
        elif self.parameters['initial_learning_rate'] > other.parameters['initial_learning_rate']: return -1
        elif self.name < other.name: return 1
        elif self.name > other.name: return -1
        else: return 0



class Result:
    def __init__(self, true_positives, false_negatives, false_positives, true_negatives):
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.true_negatives = true_negatives
        if true_positives + false_positives > 0: self.precision = float(true_positives) / (true_positives + false_positives)
        else: self.precision = float('nan')
        if true_positives + false_negatives > 0: self.recall = float(true_positives) / (true_positives + false_negatives)
        else: self.recall = float('nan')
        self.accuracy = float(true_positives + true_negatives) / (true_positives + false_negatives + false_positives + true_negatives)



class Layer:
    def __init__(self, function, input_size, output_size):
        self.function = function
        self.input_size = input_size
        self.output_size = output_size



class Architecture:
    def __init__(self, layers):
        self.layers = layers
        self.input_size = layers[0].input_size
        self.nlayers = len(layers)
        for iv in range(self.nlayers):
            if layers[iv+1].function == 'flatten':
                self.output_size = layers[iv].output_size
                break




def ParseResults(filename):
    # open the results filename
    with open(filename, 'r') as fd:
        for line in fd.readlines():
            # read the important information from the results
            if 'Merge | ' in line:
                true_positives = int(line.split()[3])
                false_negatives = int(line.split()[4])
            elif 'Split | ' in line:
                false_positives = int(line.split()[3])
                true_negatives = int(line.split()[4])

    return Result(true_positives, false_negatives, false_positives, true_negatives)



def ParseLogFile(filename):
    parameters = {}
    layers = []

    with open(filename, 'r') as fd:
        reading_parameters = False
        for line in fd.readlines():
            line = line.strip()

            if reading_parameters: 
                components = line.split(': ')
                parameters[components[0]] = components[1]
            elif not len(line): 
                reading_parameters = True
            else:
                function = line.split('_')[0]
                input_size = line.split('_')[-1].split(' -> ')[0].replace('None, ', '')
                input_size = input_size[input_size.index(' ')+1:]
                output_size = line.split('_')[-1].split(' -> ')[1].replace('None, ', '')

                layers.append(Layer(function, input_size, output_size))

    return Architecture(layers), parameters



def CNNResults(problem):
    # get all of the trained networks for this problem
    parent_directory = 'architectures/{}'.format(problem)
    network_names = os.listdir(parent_directory)

    # create a set of all of the trained networks
    networks = []

    # iterate over all networks
    for name in network_names:
        if name == 'small-train': continue
        directory = '{}/{}'.format(parent_directory, name)

        # there needs to be a log file for this to work
        logfile = glob.glob('{}/*.log'.format(directory))[0]
        result_files = sorted(glob.glob('{}/*.results'.format(directory)))

        # parse the results and logfiles
        architecture, parameters = ParseLogFile(logfile)

        results = []
        for filename in result_files:
            results.append(ParseResults(filename))
            
        networks.append(NetworkResult(name, architecture, parameters, results))

    return sorted(networks)
