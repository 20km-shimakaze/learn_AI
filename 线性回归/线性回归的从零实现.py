import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy


def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    # 均值为0，方差为1，num行，len(w)列
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size_s, features_s, labels_s):
    num_examples = len(features_s)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size_s):
        batch_indices = torch.tensor(indices[i: min(i + batch_size_s, num_examples)])
        yield features_s[batch_indices], labels_s[batch_indices]


def linreg(X, w, b):
    """线性回归"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    # y_hat预测值 y真实值
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def sgd(params, lr, batch_size):
    # 所有参数(w, b) 学习率 批量大小
    """小批量随机下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


true_w = torch.tensor([2, -3.4])
true_b = 4.3
features, labels = synthetic_data(true_w, true_b, 1000)
print('features', features[0], '\nlabel', labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
plt.show()
batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break
w = torch.normal(0, 0.01, (2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.03
# 进行3次循环
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # X和y的小批量损失
        l = loss(net(X, w, b), y)
        # loss返回的是['batch_size', 1]，没有sum
        # 求关于['w','b']的梯度
        l.sum().backward()
        # 使用参数梯度更新
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(X, w, b), y)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


