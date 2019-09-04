import torch
import torch.nn as nn

x = torch.randn(1,20)
print(x)
#print(x.shape)
self1 = nn.Linear(20,3)
#for i in range(10):
    #print(self1(x))