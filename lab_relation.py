from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np 

import torch
import torch.nn.functional as F 
import torch.optim as optim

from models import FNN
from data import load_data_relation

import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true',default=False,
                    help='Disables CUDA training')
parser.add_argument('--seed', type=int, default=233, help="Random seed")
parser.add_argument('--epochs', type=int, default=1000,
                    help="Number of training epochs")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

feature, weight, out, idx_train, idx_val, idx_test = load_data_relation()

if args.cuda:
    feature = feature.cuda()
    weight = weight.cuda()
    out = out.cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()
    idx_val = idx_val.cuda()

def train_epoch(epoch, model, optimizer, lambd) :
    model.train()
    optimizer.zero_grad()
    output = torch.tensor(0)

    output = model(feature[:,0,:])
    output = output.unsqueeze(1)
    for i in range(1, feature.shape[2]) :
        output = torch.cat((output, model(feature[:, i, :]).unsqueeze(1)), 1)

    t_weight = torch.stack((weight, weight),2)

    output = torch.mul(t_weight, output)
    output = torch.sum(output, 1)

    loss_train = F.mse_loss(output[idx_train], out[idx_train])

    loss_train.backward()
    optimizer.step()


def train(epochs, layer, lr, lambd) :
    model = FNN(feature.shape[2], out.shape[1], layer, 128)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = lambd)

    if args.cuda :
        model = model.cuda()
    
    print("Training FNN for %d layers, %f learning rate, %f lambda" %(layer, lr, lambd))

    for epoch in range(epochs) :
        train_epoch(epoch, model, optimizer, lambd)

    output = model(feature[:,0,:])
    output = output.unsqueeze(1)
    for i in range(1, feature.shape[2]) :
        output = torch.cat((output, model(feature[:, i, :]).unsqueeze(1)), 1)

    t_weight = torch.stack((weight, weight),2)

    output = torch.mul(t_weight, output)
    output = torch.sum(output, 1)

    loss_train = F.mse_loss(output[idx_train], out[idx_train])

    loss_val = F.mse_loss(output[idx_val], out[idx_val])

    print("Result for %d layers, %f learning rate, %f lambda" %(layer, lr, lambd))
    print('loss_val: {:.4f}'.format(loss_val.item()))

    return model, output, loss_val

lrs = [0.0001, 0.001, 0.01, 0.1]
lambdas = [0.000001, 0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1]
layers = [2,3,4,5,6]


min_loss = 100000000
min_lr = -1
for lr in lrs :
    model, output, cur_loss = train(args.epochs, 2, lr, 0.1)
    if(cur_loss < min_loss) :
        min_loss = cur_loss
        min_lr = lr


min_loss = 100000000
min_layer = -1
for layer in layers :
    model, output, cur_loss = train(args.epochs, layer, min_lr, 0.1)
    if(cur_loss < min_loss) :

        min_loss = cur_loss
        min_layer = layer

min_loss = 100000000
min_lambda = -1
for lambd in lambdas :
    model, output, cur_loss = train(args.epochs, min_layer, min_lr, lambd)
    if(cur_loss < min_loss) :
        min_loss = cur_loss
        min_lambda = lambd


model,output, loss = train(args.epochs, min_layer, min_lr, min_lambda)
output = model(feature[:,0,:])
output = output.unsqueeze(1)
for i in range(1, feature.shape[2]) :
    output = torch.cat((output, model(feature[:, i, :]).unsqueeze(1)), 1)

t_weight = torch.stack((weight, weight),2)

output = torch.mul(t_weight, output)
output = torch.sum(output, 1)
 
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
