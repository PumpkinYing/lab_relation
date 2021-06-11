from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np 

import torch
import torch.nn as nn
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
parser.add_argument('--system', type=str, help='System of dataset')
parser.add_argument('--sample_size', type=int, help="Sample size")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

feature, out = load_data("data/"+args.system+"_AllNumeric.csv")

if args.cuda:
    feature = feature.cuda()
    out = out.cuda()

def weight_init(m) :
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

def relative_loss(output, out) :
    loss = torch.div(torch.abs(output-out), out)
    loss = torch.sum(loss, 0)
    loss = loss*100/output.shape[0]
    return loss.item()

def print_pic(lr, layer, loss, lambd, output) :
    plt.figure()
    mx_idx = output.shape[0]
    plt.plot(range(mx_idx), output.detach().cpu().numpy(), label='output')
    plt.plot(range(mx_idx), out.detach().cpu().numpy(), label='true')
    plt.legend(loc=3)
    plt.savefig('./pics/'+args.system+'_%f_lr_%d_layer_%f_loss_%f_lambda.png'%(lr, layer, loss, lambd))

def train_epoch(epoch, model, optimizer, lambd, idx_train, idx_val):
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

def train(epochs, layer, lr, lambd, idx_train, idx_val):
    model = FNN(feature.shape[1], out.shape[1], layer, 128)
    model.apply(weight_init)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = lambd)

    if args.cuda:
        model = model.cuda()

    print("Training FNN for %d layers, %f learning rate, %f lambda" %(layer, lr, lambd))

    for epoch in range(epochs) :
        train_epoch(epoch, model, optimizer, lambd, idx_train, idx_val)

    output = model(feature)
    loss_val = F.mse_loss(output[idx_val], out[idx_val])

    print("Result for %d layers, %f learning rate, %f lambda" %(layer, lr, lambd))
    print('loss_val: {:.4f}'.format(loss_val.item()))
    
    return output, loss_val


def experiment(seed):
    sample_size = args.sample_size

    (n, m) = feature.shape
    np.random.seed(seed)
    permutation = np.random.permutation(n)
    idx_sample = permutation[0:sample_size]
    split_pos = int(np.ceil(sample_size*2.0/3))
    idx_train = idx_sample[0:split_pos]
    idx_val = idx_sample[split_pos:]
    
    
    lrs = np.logspace(np.log10(0.0001), np.log10(0.1), 4)
    lambdas = np.logspace(-2, np.log10(1000), 30)
    layers = range(2,15)

    MAX_LOSS = 100000000000

    # Pick the layer with smallest val loss
    best_layer = -1
    min_loss = MAX_LOSS
    for layer in layers :
        min_layer_loss = MAX_LOSS
        for lr in lrs :
            output, loss_val = train(args.epochs, layer, lr, 0.1, idx_train, idx_val)
            if loss_val < min_layer_loss :
                min_layer_loss = loss_val

        if min_layer_loss < min_loss :
            min_loss = min_layer_loss
            best_layer = layer

    best_layer = best_layer

    best_lr = -1
    min_loss = MAX_LOSS
    for lr in lrs :
        output, loss_val = train(args.epochs, best_layer, lr, 0.1, idx_train, idx_val)
        if loss_val < min_loss :
            min_loss = loss_val
            best_lr = lr

    best_layer = best_layer+5
    best_lambda = -1
    min_loss = MAX_LOSS
    for lambd in lambdas :
        output, loss_val = train(args.epochs, best_layer, best_lr, lambd, idx_train, idx_val)
        if(loss_val < min_loss) :
            min_loss = loss_val
            best_lambda = lambd

    output, loss_val = train(args.epoches, best_layer, best_lr, best_lambda, idx_train, idx_val)

    total_rmse = relative_loss(output, out)
    print_pic(best_lr, best_layer, total_rmse, best_lambda, output)




    # model, loss = train(args.epochs, min_layer+5, min_lr, min_lambda, idx_train, idx_val)
    # output = model(feature)
    # print("final args: layer: %d, lr: %f, lambda: %f" %(min_layer, min_lr, min_lambda))
    # print("final loss: %f" %(loss))
