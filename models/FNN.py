import torch 
import torch.nn.functional as F 

class FNN(torch.nn.Module) :
    def __init__(self, in_features, out_features, layer_num, node_num):
        super(FNN, self).__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(in_features, node_num))
        for num in range(layer_num-1):
            self.layers.append(torch.nn.Linear(node_num, node_num))
        self.layers.append(torch.nn.Linear(node_num, out_features))

        self.layers = torch.nn.ModuleList(self.layers)

    
    def forward(self, x):
        for layer in self.layers :
            x = layer(x)
        return F.relu(x)