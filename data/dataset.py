import numpy as np 
import torch

def load_data(path="./data/") :
    print("loading dataset")

    matrix = np.loadtxt(path+"matrix.txt", delimiter=' ')
    feature = np.loadtxt(path+"feature_normalize.txt", delimiter=' ')
    out = np.loadtxt(path+"out.txt", delimiter=' ')

    feature = torch.FloatTensor(normalize(feature))
    matrix = torch.FloatTensor(matrix+np.eye(matrix.shape[0]))
    matrix = torch.FloatTensor(normalize(matrix))
    out = torch.FloatTensor(out)

    id_train = range(500)
    id_val = range(500,1000)
    id_test = range(1100,1600)

    id_train = torch.LongTensor(id_train)
    id_val = torch.LongTensor(id_val)
    id_test = torch.LongTensor(id_test)

    return matrix, feature, out, id_train, id_val, id_test

def load_data_relation(system, path="./data/") :

    train_num = {}
    train_num["Apache"] = 80
    train_num["Self"] = 500 
    train_num["BDBC"] = 800

    print("loading dataset")

    feature = np.loadtxt(path+system+"_train.txt", delimiter=' ')
    feature = feature.reshape((int(feature.shape[0]/10),10,feature.shape[1]), order = "C")
    weight = np.loadtxt(path+system+"_weight.txt", delimiter=' ')
    out = np.loadtxt(path+system+"_out.txt", delimiter=' ')

    feature = torch.FloatTensor(feature)
    weight = torch.FloatTensor(weight)
    out = torch.FloatTensor(out)

    id_train = range(train_num[system])
    id_val = range(train_num[system], 2*train_num[system])
    id_test = range(1000, out.shape[0])

    id_train = torch.LongTensor(id_train)
    id_val = torch.LongTensor(id_val)
    id_test = torch.LongTensor(id_test)

    return feature, weight, out, id_train, id_val, id_test

def normalize(mx) :
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx