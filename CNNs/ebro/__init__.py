import os
os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'
from baseline import *
from forward import *
from generate_features import *
from train import *

