import torch
'''V_data=[[1.],[2],[3]]
V=torch.Tensor(V_data)
print(V)'''
'''x_1 = torch.randn((2,5))
y_1 = torch.randn((3,5))
print(x_1)
print(y_1)
z_1 = torch.cat([x_1, y_1])
print(z_1)'''
'''x_2=torch.randn((2,2))
y_2=torch.randn((2,3))
print(x_2)
print(y_2)
z_2=torch.cat([x_2, y_2],1)
print(z_2)'''
'''x = torch.randn(2,3,4)
print(x)
x.view((2,12))
print(x)'''

'''x = torch.tensor([1.,2,3], requires_grad=True)
y = torch.tensor([2.,3,4], requires_grad=True)
z=x+y
print(z)
print(z.grad_fn)
s=z.sum()
print(s)
print("blablalbla")
s.backward()
print(x.grad)'''

x=torch.randn(2,2)
y=torch.randn(2,2)
print(x.requires_grad, y.requires_grad)
z=x+y
print(z.grad_fn)

x=x.requires_grad_()
y=y.requires_grad_()
print(x,y)
z=x+y
s=z.sum()
s.backward()
print(x.grad)

