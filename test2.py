import torch
import torch.nn as nn
'''
a = torch.Tensor([[-2.9520, -2.7817, -2.8526, -2.8429, -2.8961, -2.9682, -2.7748, -2.8996,
         -3.0847, -2.9231, -2.8625, -2.8539, -2.9629, -2.8704, -2.7592, -2.9407,
         -2.8607, -2.9989]])
print(a)
criterion = nn.NLLLoss()
'''
'''
a = torch.Tensor([[1,2,3], [4, 5, 6], [7, 8, 9]])
print(a)
a=a.sum(dim=0)
print(a)
a=a.sum(dim=0)
print(a)#dim=0 一维表示行 二维表示列？ dim=0表示的是最外面的一项
'''
'''
b = torch.Tensor([[[1,2,3], [4, 5, 6], [7, 8, 9]], [[1,2,3], [4, 5, 6], [7, 8, 9]]])
print(b)
print(b.sum(dim=0))
'''
'''
tensor([[ 2.,  4.,  6.],
        [ 8., 10., 12.],
        [14., 16., 18.]])
'''
'''
'''