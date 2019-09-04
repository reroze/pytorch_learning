#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


CONTEXT_SIZE = 2#左边两个，右边两个
EMBEDDING_DIM = 10
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

#print(raw_text)
vocab =set(raw_text)#将列表变成字典 不过顺序会变 而且是随机 每次都不一样
vocab_size = len(raw_text)
#print(vocab)
#print(enumerate(vocab))
word_to_ix = {word : i for i, word in enumerate(vocab)} #for i, word in enumerate(vocab) 是指将vocab里的单词索引出来
#print(word_to_ix)制作word_to_ix
data = []
for i in range(2, len(raw_text)-2):
    context = [raw_text[i-2], raw_text[i-1], raw_text[i+1], raw_text[i+2]]
    target = raw_text[i]
    data.append((context, target))
#print(data)

class CBOW(nn.Module):

    def __init__(self, vocab_size, context_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)#前面是确定模型的一些参数
        self.linear1 = nn.Linear(context_size*2*embedding_dim, 256)
        self.linear2 = nn.Linear(256, vocab_size)



    def forward(self, inputs):#这一步大剑魔性的前向传播 inputs为对应输入的单词的序号
        embeds = self.embeddings(inputs).view(1, -1)#这里使用了view来代替下面的make_context_vector 将序列号变成对应长度的初始词向量 并拼接到一个向量里
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        out = F.log_softmax(out, dim=1)
        return out


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return  torch.tensor(idxs, dtype=torch.long)

losses = []
loss_function=nn.NLLLoss()
model = CBOW(vocab_size, CONTEXT_SIZE, EMBEDDING_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in data:
        context_idxs =make_context_vector(context, word_to_ix)#将单词变成对应的序列号
        model.zero_grad()
        log_probs1 = model(context_idxs)
        loss = loss_function(log_probs1, torch.tensor([word_to_ix[target]], dtype=torch.long))
        total_loss+=loss
        loss.backward()
        optimizer.step()
    losses.append(total_loss)

print(losses)


