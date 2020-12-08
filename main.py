from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np 

import torch
import torch.nn.functional as F 
import torch.optim as optim

from models import FNN
from data import load_data

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true',default=False,
                    help='Disables CUDA training')
parser.add_argument('--seed', type=int, default=233, help="Random seed")
parser.add_argument('--epochs', type=int, default=30,
                    help="Number of training epochs")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

matrix, feature, out, id_train, id_val, id_test = load_data()

if args.cuda:
    model.cuda()
    feature = feature.cuda()
    matrix = matrix.cuda()
    out = out.cuda()
    idx_train = id_train.cuda()
    idx_test = id_test.cuda()
    idx_val = id_val.cuda()

print(feature.shape)

def train(epoch, layer, lr, lambd):
    model = FNN()