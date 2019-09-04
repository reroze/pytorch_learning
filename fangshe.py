import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#lin=nn.Linear(5,3)

#data=torch.randn(2,5)
#print(lin(data))
#print(data)
#print(F.relu(data))

data=torch.randn(5)
print(data)
soft_data=F.softmax(data,dim=0)
print(soft_data)
print(F.log_softmax(data,dim=0))