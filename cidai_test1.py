import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BOWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BOWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)


    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)

model = BOWClassifier(2, 4)

for param in model.parameters():
    print(param)#先是一个2行4列的张量 后是一个1行2列的张量 w*x 前面那个对应着的是每个x对应第一种和第二种的4个权重w 后面那个是偏置项b



x1 = [1, 2, 3, 4]#对应的是0.0262和-1.3248
x2 = [1, 3, 5, 7]
x3 = [1, 5, 9, 13]
x4 = [2, 4, 6, 8]
x5 = [2, 6, 10, 14]
y1 = 0
y2 = 0
y3 = 0
y4 = 1
y5 = 1#每个类的标签要>=0且<n_class即class类的数目
x = [x1, x2, x3, x4, x5]
y = [y1, y2, y3, y4, y5]
x_ = x.copy()
y_ = y.copy()
print("y_:",y_)

li = nn.Linear(4,2)
x1 = torch.Tensor(x1)
#print(F.log_softmax(li(x1), dim=0))

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for i in range(5):
        model.zero_grad()
        target = torch.LongTensor([y_[i]])
        print("target:",target)
        x_ = torch.Tensor(x[i])
        x_ = x_.view(1, -1)
        print("x_:", x_)
        log_probs = model(x_)
        print("log_probs", log_probs)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
    if(epoch % 10 ==0):
        print(next(model.parameters())[1])