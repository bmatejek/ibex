import struct

import numpy as np

from ibex.utilities import dataIO



# read ground truth
def SkeletonGroundTruth(prefix, cutoff=500):
    directory = 'benchmarks/skeleton/{}'.format(prefix)

    for iv in range(cutoff):
        filename = '{}/example-{}.pts'.format(directory, iv)

        with open(filename, 'rb') as fd:
            npoints, = struct.unpack('q', fd.read(8))
            
            for ip in range(npoints):
                iz, iy, ix, = struct.unpack('qqq', fd.read(24))



# find the endpoints





# evaluate the methods
def EvaluateSkeletons(prefix, cutoff=500, resolution=(100, 100, 100)):
    SkeletonGroundTruth(prefix, cutoff)




# find skeleton benchmark information
def GenerateExamples(prefix, cutoff=500):
    gold = dataIO.ReadGoldData(prefix)
    labels, counts = np.unique(gold, return_counts=True)
    if labels[0] == 0: cutoff += 1

    filename = 'benchmarks/skeleton/{}-skeleton-benchmark-examples.bin'.format(prefix)
    with open(filename, 'wb') as fd:
        fd.write(struct.pack('q', cutoff))
        for ie, (count, label) in enumerate(sorted(zip(counts, labels), reverse=True)):
            if not label: continue
            # don't include more than cutoff examples
            if ie == cutoff: break
            fd.write(struct.pack('q', label))