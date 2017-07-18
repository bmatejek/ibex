import sys
from ibex.geometry import ibbox
from ibex.utilities.constants import *

class MetaData:
    def __init__(self, prefix):
        # open the meta data txt file
        filename = 'meta_data/{}.meta'.format(prefix)
        with open(filename, 'r') as fd:
            lines = fd.readlines()

            for ix in range(0, len(lines), 2):
                # get the comment and the corresponding value
                comment = lines[ix].strip()
                value = lines[ix + 1].strip()

                if comment == '# resolution in nm':
                    # separate into individual dimensions
                    samples = value.split('x')
                    self.resolution = (int(samples[2]), int(samples[1]), int(samples[0]))
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
                    self.bounding_box = ibbox.IBBox(mins, maxs)
                else:
                    sys.stderr.write('Unrecognized attribute in {}: {}\n'.format(prefix, comment))
                    sys.exit()


    def BBox(self):
        return self.bounding_box

    def Resolution(self):
        return self.resolution

    def SegmentationFilename(self):
        if self.rhoana_filename == None:
            return 'rhoana/{}_rhoana.h5'.format(prefix), 'main'
        else:
            return self.rhoana_filename.split()[0], self.rhoana_filename.split()[1]