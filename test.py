import torch
import torch.nn as nn

m = nn.Linear(9, 2) 
input = torch.randn(1000, 10, 9) #dimensions: i item   j with xx item   k xx element
output = torch.tensor(0)

output = m(input[:,0,:])
output = output.unsqueeze(1)
for i in range(1,10) :
    output = torch.cat((output,m(input[:,i,:]).unsqueeze(1)),1)
    print(output.shape)

weight = torch.randn(1000,10)
weight = torch.stack((weight, weight), 2)
print(weight.shape)

output = torch.mul(weight, output)

output = torch.sum(output, 1)
print(output.shape)