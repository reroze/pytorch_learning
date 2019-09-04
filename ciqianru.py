import torch
import torch.nn as nn
import torch.nn.functional  as F
import torch.optim as optim

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)
#print("embeds:", embeds)
lookup_tensor = torch.tensor([word_to_ix['hello'], word_to_ix['world']], dtype=torch.long)#可以同时对一个列表里的多个元素进行词向量化
#print(lookup_tensor)
hello_embeds = embeds(lookup_tensor)#将原来的键值对变成了embedding中的词向量
print(hello_embeds)