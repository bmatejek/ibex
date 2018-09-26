from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from ibex.utilities import dataIO
from ibex.transforms import seg2gold
from ibex.evaluation import classification

import numpy as np



nfeatures = 3


def ExtractFeatures(prefix):
    # read in the datasets
    segmentation = dataIO.ReadSegmentationData(prefix)
    gold = dataIO.ReadGoldData(prefix)
    resolution = dataIO.Resolution(prefix)

    # find the mapping from segmentation to gold
    seg2gold_mapping = seg2gold.Mapping(segmentation, gold)

    # read in the skeletons
    skeletons = dataIO.ReadSkeletons(prefix)

    nskeletons = 0
    for ik in range(skeletons.NSkeletons()):
        skeleton = skeletons.KthSkeleton(ik)
        if skeleton.NPoints() < 10: continue

        label = skeleton.label
        if not seg2gold_mapping[label]: continue

        nskeletons += 1


    # create feature vector and label vector for learning
    features = np.zeros((nskeletons, nfeatures), dtype=np.float32)
    labels = np.zeros((nskeletons), dtype=np.uint8)

    nskeletons = 0
    for ik in range(skeletons.NSkeletons()):
        skeleton = skeletons.KthSkeleton(ik)
        if skeleton.NPoints() < 10: continue

        label = skeleton.label
        if not seg2gold_mapping[label]: continue

        joints = skeleton.WorldJoints2Array(resolution)

        pca = PCA(n_components=3)        
        pca.fit(joints)

        features[nskeletons,0] = pca.explained_variance_ratio_[0]
        features[nskeletons,1] = pca.explained_variance_ratio_[1]
        features[nskeletons,2] = pca.explained_variance_ratio_[2]
        # if pca.explained_variance_ratio_[1] == 0.0:
        #     features[nskeletons,3] = 10e10
        # else:
        #     features[nskeletons,3] = pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]
        # if pca.explained_variance_ratio_[2] == 0.0:
        #     features[nskeletons,4] = 10e10
        # else:
        #     features[nskeletons,4] = pca.explained_variance_ratio_[1] / pca.explained_variance_ratio_[2]
        # if pca.explained_variance_ratio_[2] == 0.0:
        #     features[nskeletons,5] = 10e10
        # else:
        #     features[nskeletons,5] = pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[2]
        # features[nskeletons,0] = skeleton.NPoints()
        # features[nskeletons,1] = skeleton.NEndpoints()
        # features[nskeletons,2] = skeleton.NJoints()

        labels[nskeletons] = (seg2gold_mapping[label] == -1)

        nskeletons += 1

    return features, labels



def DetectMergeErrors(train_prefixes, test_prefixes):
    train_features = np.zeros((0, nfeatures), dtype=np.float32)
    train_labels = np.zeros((0), dtype=np.float32)

    for train_prefix in train_prefixes:
        print train_prefix
        features, labels = ExtractFeatures(train_prefix)
        train_features = np.concatenate((train_features, features))
        train_labels = np.concatenate((train_labels, labels))
    
    clf = RandomForestClassifier(class_weight='balanced')
    clf.fit(train_features, train_labels)
    
    for test_prefix in test_prefixes:
        print test_prefix
        test_features, test_labels = ExtractFeatures(test_prefix)

        test_probabilities = clf.predict(test_features)
        test_predictions = classification.Prob2Pred(test_probabilities)    

        classification.PrecisionAndRecall(test_labels, test_predictions)
