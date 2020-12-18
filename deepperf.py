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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true',default=False,
                    help='Disables CUDA training')
parser.add_argument('--seed', type=int, default=233, help="Random seed")
parser.add_argument('--epochs', type=int, default=500,
                    help="Number of training epochs")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

matrix, feature, out, id_train, id_val, id_test = load_data()

if args.cuda:
    feature = feature.cuda()
    matrix = matrix.cuda()
    out = out.cuda()
    idx_train = id_train.cuda()
    idx_test = id_test.cuda()
    idx_val = id_val.cuda()

def train_epoch(epoch, model, optimizer, lambd):
    model.train()
    optimizer.zero_grad()
    output = model(feature)

    loss_train = F.mse_loss(output[idx_train], out[idx_train])
    # for param in model.parameters() :
    #     for ps in param :
    #         for p in ps :
    #             loss_train = loss_train + lambd*p
    #     break
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(feature)

    loss_val = F.mse_loss(output[idx_val], out[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()))

def train(epochs, layer, lr, lambd):
    model = FNN(feature.shape[1], out.shape[1], layer, 128)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = lambd)

    if args.cuda:
        model = model.cuda()

    print("Training FNN for %d layers, %f learning rate, %f lambda" %(layer, lr, lambd))

    for epoch in range(epochs) :
        train_epoch(epoch, model, optimizer, lambd)

    output = model(feature)
    loss_val = F.mse_loss(output[idx_val], out[idx_val])

    print("Result for %d layers, %f learning rate, %f lambda" %(layer, lr, lambd))
    print('loss_val: {:.4f}'.format(loss_val.item()))
    
    return model, loss_val


lrs = [0.0001, 0.001, 0.01, 0.1]
lambdas = [0.000001, 0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1]
layers = [2,3,4,5,6]

min_loss = 100000000
min_lr = -1
for lr in lrs :
    model, cur_loss = train(args.epochs, 2, lr, 0.1)
    if(cur_loss < min_loss) :
        min_loss = cur_loss
        min_lr = lr

min_loss = 100000000
min_layer = -1
for layer in layers :
    model, cur_loss = train(args.epochs, layer, min_lr, 0.1)
    if(cur_loss < min_loss) :
        min_loss = cur_loss
        min_layer = layer

min_loss = 100000000
min_lambda = -1
for lambd in lambdas :
    model, cur_loss = train(args.epochs, layer, min_lr, lambd)
    if(cur_loss < min_loss) :
        min_loss = cur_loss
        min_lambda = lambd

model, loss = train(args.epochs, min_layer, min_lr, min_lambda)
output = model(feature)
print("final args: layer: %d, lr: %f, lambda: %f" %(min_layer, min_lr, min_lambda))
print("final loss: %f" %(loss))

plt.figure()
plt.subplot(2,1,1)
mx_idx = output.shape[0]
plt.plot(range(mx_idx), output.detach().cpu().numpy()[:,0], label='output')
plt.plot(range(mx_idx), out.detach().cpu().numpy()[:,0], label='true')
plt.legend(loc=3)
plt.subplot(2,1,2)
plt.plot(range(mx_idx), output.detach().cpu().numpy()[:,1], label='output')
plt.plot(range(mx_idx), out.detach().cpu().numpy()[:,1], label='true')
plt.legend(loc=3)
plt.savefig('./pics/'+'result'+'.png')
