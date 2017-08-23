import os
os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'

# import files from this directory
from train import Train
from forward import Forward
from generate_features import GenerateFeatures
from feature_analysis import FeatureAnalysis