import torch 
import torch.nn.functional as F 

class FNN(torch.nn.Module) :
    def __init__(self, in_features, out_features, layer_num, Lambda):
        super(FNN, self).__init__()
        self.fc = []
        self.fc.append(nn.Linear(in_features, 128))
        for num in range(layer_num-1):
            self.fc.append(nn.Linear(128, 128))
        self.fc.append(nn.Linear(128,out_features))
    
    def forward(self, x):
        for fc in self.fc :
            x = fc(x)
        return x