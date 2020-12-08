import numpy as np 
import torch

def load_data(path="./data/") :
    print("loading dataset")

    matrix = np.loadtxt(path+"matrix.txt", delimiter=' ')
    feature = np.loadtxt(path+"feature.txt", delimiter=' ')
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

def normalize(mx) :
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx