import os
os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'
from train import *
from forward import *
