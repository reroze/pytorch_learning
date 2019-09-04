#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

CONTEXT_SIZE = 2
#CONTEXT_SIZE = 3#对应为3的情况
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigrams = [([test_sentence[i], test_sentence[i+1]], test_sentence[i+2]) for i in range(len(test_sentence) - 2)]#此时n为2， 所以要变成2+1的形式
#trigrams = [([test_sentence[i], test_sentence[i+1], test_sentence[i+2]], test_sentence[i+3]) for i in range(len(test_sentence) - 3)]#此时n为3， 所以要变成2+1的形式
#print(trigrams[0][0][0])

#print(trigrams)
#print(trigrams[:3])
#print(test_sentence)
vocab = set(test_sentence) #将列表变成集合 顺序会变 每次的顺序是随机的
#print(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}#将对应的字典按顺序变为 索引
word_list = list(word_to_ix)
#print(word_to_ix)


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)#原来的单词的个数将其映射到embedding_dim维度的词向量 vocab_size应该指的是n_gram里的n #定义词嵌入词向量生成模型
        #print(self.embeddings)
        self.linear1 = nn.Linear(context_size*embedding_dim, 128)#每次进行训练的只有两个单词的词向量 隐含层为128个神经元
        self.linear2 = nn.Linear(128, vocab_size)#最后医院来的词的个数作为结果的类数


    def forward(self, inputs): #question2: what is the origin of the input
        embeds = self.embeddings(inputs).view((1, -1))#对于输入的inputs，生成对应的词向量
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)#一个【】中的便称为0维，两个便称为1维 比如[[1, 2, 3],[2,3,4] sum dim=0->[3,5,7] sum dim=1->[[6],[9]]
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)#embedding_dim=10, context_size=2
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(100):
    total_loss = 0
    for context, target in trigrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype = torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss
    losses.append(total_loss)
print(losses)

good = 0
for time in range(100):
    test_ix = random.randint(0, 100)
    #print(test_ix)

    test_context = trigrams[test_ix][0]
    test_target = trigrams[test_ix][1]
    #print(test_context, test_target)
    context_idxs = torch.tensor([word_to_ix[w] for w in test_context], dtype = torch.long)
    log_probs = model(context_idxs)
    #print(log_probs)
    #print(log_probs.topk(3, dim=1))#返回前3个
    #print(log_probs.argmax())
    probis = log_probs.topk(3, dim=1)[1]
    #print(probis)
    target_ix = word_to_ix[test_target]
    #print(word_to_ix[test_target])
    probility = []
    if target_ix in probis:
        good+=1
    for ex in probis[0]:
        #print(ex)
        #print(word_list[int(ex)])
        probility.append(word_list[int(ex)])
    print("probs:", probility)
    print("real:", test_target)
print("good:", good)

