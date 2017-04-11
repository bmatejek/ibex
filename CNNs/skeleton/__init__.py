import os
os.environ['THEANO_FLAGS'] = 'device=cuda,floatX=float32'
from generate_features import *
from train import *
from forward import *
