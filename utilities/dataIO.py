import h5py
import numpy as np

def ReadMeta(prefix):
    filename = 'meta_data/' + prefix + '.meta'

    with open(filename, 'r') as fd:
        meta_data = fd.readlines()

        resolutions = meta_data[1].split('x')

        return (int(resolutions[2]), int(resolutions[1]), int(resolutions[0]))

def ReadH5File(filename, dataset):
    with h5py.File(filename, 'r') as hf:
        data = np.array(hf[dataset])

    return data
        
