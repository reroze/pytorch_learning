#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


lstm = nn.LSTM(3, 4)
inputs = [torch.randn(1, 3) for _ in range(5)]
print("inputs:", inputs)

#a = torch.randn(2) :tensor([0.3322, 0.7202])
#print(a)

hidden = (torch.randn(1, 1, 4), torch.randn(1, 1, 4))
#print(hidden)

for i in inputs:
    #print("i:", i)
    #print("i`:", i.view(1, 1, -1)) :i`: tensor([[[ 0.8641, -1.1995,  1.3710]]]) #分开运算
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    print(out)#此时的out为1*4

inputs = torch.cat(inputs).view(len(inputs), 1, -1)#接到一个整合里了
#print("inputs`:", inputs)
hidden = (torch.randn(1, 1, 4), torch.randn(1, 1, 4 ))
out, hidden = lstm(inputs, hidden)
#print(out)#此时的out为4*1*4
#print(hidden)