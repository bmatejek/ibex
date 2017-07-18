import os
os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'
from generate_features import *