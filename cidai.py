import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


data = [("me gusta comer en la cafeteria".split(), "SPANISH"),#data对应的类型是前面是一句话的组成成分，后面是该句话的语种
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]
test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]
word_to_ix = {}


for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)#实时的更新对应的新单词的序列号
print(word_to_ix)#{'me': 0, 'gusta': 1, 'comer': 2, 'en': 3, 'la': 4, 'cafeteria': 5, 'Give': 6, 'it': 7, 'to': 8, 'No': 9, 'creo': 10, 'que': 11, 'sea': 12, 'una': 13, 'buena': 14, 'idea': 15, 'is...

VOCAB_SIZE = len(word_to_ix)
#print(VOCAB_SIZE)#此处为26个
NUM_LABELS = 2#num_labels是什么？ 两类，分别为English和Spanish

class BOWClassifier(nn.Module):#BOw分类器
    def __init__(self, num_labels, vocab_size):
        super(BOWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size,num_labels)#从变量的维数到最后分类的维数


    def forward(self, bow_vec):
        print("bow_vec:",bow_vec)
        print("softmax:",F.log_softmax(self.linear(bow_vec), dim=1))
        return F.log_softmax(self.linear(bow_vec), dim=1)#按列进行softmax。。。。

def make_bow_vector(sentence, word_to_ix):#生成对应的词向量
    vec = torch.zeros(len(word_to_ix))
    #print("vec:", vec)
    for word in sentence:
        ix = word_to_ix[word]
        vec[ix] += 1#统计句子中每个单词出现的个数
    #print("vec:", vec.view(1, -1))
    return vec.view(1, -1)#shape成一维的


def make_target(label, label_to_ix):#返回对应的标签
    return torch.LongTensor([label_to_ix[label]])


model = BOWClassifier(NUM_LABELS, VOCAB_SIZE)#VOCAB_SIZE=26, NUM_LABELS=2#定义对应的模型，类似tf.matmul

for param in model.parameters():
    print(param)#刚好有2*26个 对应着bow——vec的维数

'''with torch.no_grad():
    sample = data[0]#SAMPLE是data的第一个数据
    bow_vector = make_bow_vector(sample[0], word_to_ix)
    log_probs = model(bow_vector)
    print(log_probs)


'''
label_to_ix = {"SPANISH": 0, "ENGLISH": 1}#定义第一个参数组和第二个参数组分别为西班牙语和英语


with torch.no_grad():
    for instance, label in test_data:
        bow_vector = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vector)
        print(log_probs)

print(next(model.parameters())[:, word_to_ix['creo']])#单词creo对应的两个参数 的初值

loss_function = nn.NLLLoss()#定义算是函数
optimizer = optim.SGD(model.parameters(), lr=0.1)#定义优化器



for epoch in range(100):#epoch是什么？ 循环次数？
    for instance, label in data:
        model.zero_grad()#梯度归零？
        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)#返回对应的状态 [0] 或[1]
        log_probs = model(bow_vec)
        print("log_probs:", log_probs)
        print("target", target)
        loss = loss_function(log_probs, target)
        loss.backward()#反向传播 减小loss
        optimizer.step()#梯度优化 对应的参数
    if(epoch % 10 == 0 ):
        print(next(model.parameters())[:, word_to_ix['creo']])


with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

print(next(model.parameters())[:, word_to_ix['creo']])


