import sys
from ibex.geometry import ib3shapes
from ibex.utilities.constants import *

class MetaData:
    def __init__(self, prefix):
        # initialize the prefix variable
        self.prefix = prefix

        # backwards compatability variable defaults
        self.affinity_filename = None
        self.boundary_filename = None
        self.gold_filename = None
        self.image_filename = None
        self.mask_filename = None
        self.rhoana_filename = None
        self.synapse = None

        # open the meta data txt file
        filename = 'meta/{}.meta'.format(prefix)
        with open(filename, 'r') as fd:
            lines = fd.readlines()

            for ix in range(0, len(lines), 2):
                # get the comment and the corresponding value
                comment = lines[ix].strip()
                value = lines[ix + 1].strip()

                if comment == '# resolution in nm':
                    # separate into individual dimensions
                    samples = value.split('x')
                    self.resolution = (float(samples[2]), float(samples[1]), float(samples[0]))
                elif comment == '# affinity filename':
                    self.affinity_filename = value
                elif comment == '# boundary filename':
                    self.boundary_filename = value
                elif comment == '# gold filename':
                    self.gold_filename = value
                elif comment == '# image filename':
                    self.image_filename = value
                elif comment == '# mask filename':
                    self.mask_filename = value
                elif comment == '# rhoana filename':
                    self.rhoana_filename = value
                elif comment == '# synapse filename':
                    self.synapse_filename = value
                elif comment == '# world bounding box':
                    if value == 'None': 
                        self.bounding_box = ib3shapes.IBBox((0, 0, 0), (-1, -1, -1))
                        continue
                    # separate into minimums and maximums
                    minmax = value.split('-')
                    mins = (minmax[0].strip('()')).split(',')
                    maxs = (minmax[1].strip('()')).split(',')

                    # split into individual mins
                    for iv in range(NDIMS):
                        mins[iv] = int(mins[iv])
                        maxs[iv] = int(maxs[iv])

                    # swap x and z
                    tmp = mins[0]
                    mins[0] = mins[2]
                    mins[2] = tmp

                    tmp = maxs[0]
                    maxs[0] = maxs[2]
                    maxs[2] = tmp

                    # save the bounding box
                    self.bounding_box = ib3shapes.IBBox(mins, maxs)
                else:
                    sys.stderr.write('Unrecognized attribute in {}: {}\n'.format(prefix, comment))
                    sys.exit()


    def WorldBBox(self):
        return self.bounding_box

    def Resolution(self):
        return self.resolution

    def GoldFilename(self):
        if self.gold_filename == None:
            return 'gold/{}_gold.h5'.format(self.prefix), 'main'
        else:
            return self.gold_filename.split()[0], self.gold_filename.split()[1]

    def ImageFilename(self):
        if self.image_filename == None:
            return 'images/{}_image.h5'.format(self.prefix), 'main'
        else:
            return self.image_filename.split()[0], self.image_filename.split()[1]

    def SegmentationFilename(self):
        if self.rhoana_filename == None:
            return 'rhoana/{}_rhoana.h5'.format(self.prefix), 'main'
        else:
            return self.rhoana_filename.split()[0], self.rhoana_filename.split()[1]

    def AffinityFilename(self):
        if self.affinity_filename == None:
            return 'affinities/{}_affinities.h5'.format(self.prefix), 'main'
        else:
            return self.affinity_filename.split()[0], self.affinity_filename.split()[1]