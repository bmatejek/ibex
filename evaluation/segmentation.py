from ibex.transforms import seg2seg


vi_threshold = 0.02
anisotropic = True
dilate1 = 1
dilatebase = 1
filtersize = 100
relabel1 = True
relabelbase = True



def VariationOfInformation(segmentation, gold):
    # remove small connected components
    segmentation = seg2seg.RemoveSmallConnectedComponents(segmentation, min_size=filtersize)
