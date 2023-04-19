from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
from torch import nn


def load_array(data_array, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

# 定义输入维度为2，输出维度为1
net = nn.Sequential(nn.Linear(2, 1))

# 设w为均值为0， 方差0.01
net[0].weight.data.normal_(0, 0.01)
# 设b=0
net[0].bias.data.fill_(0)

# 设计loss函数
loss = nn.MSELoss()
# 实例化SGD（小批量随机梯度下降）
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
