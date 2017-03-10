import argparse
import numpy as np
import h5py

def label_map(val, mapping):
    return mapping[val]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('segmentation', type=str, help='filename for segmentation dataset')

    args = parser.parse_args()

    with h5py.File(args.segmentation, 'r') as hf:
        segmentation = np.array(hf['main'], dtype=np.uint64)

    unique = np.unique(segmentation)

    mapping = {}
    for iv, label in enumerate(unique):
        mapping[label] = iv

    import time
    start_time = time.time()

    vectorized = np.vectorize(label_map)
    vectorized(segmentation, mapping)

    # (zres, yres, xres) = segmentation.shape
    # for iz in range(0, zres):
    #     print iz
    #     for iy in range(0, yres):
    #         for ix in range(0, xres):
    #             segmentation[iz,iy,ix] = mapping[segmentation[iz,iy,ix]]

    print time.time() - start_time

if __name__ == '__main__':
    main()