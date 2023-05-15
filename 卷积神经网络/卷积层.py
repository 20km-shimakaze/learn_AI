import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X, K):
    """矩阵×卷积核"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1), device=X.device)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# 卷积核
kernel = torch.tensor([[1., -1.]], device=d2l.try_gpu())

X = torch.ones((6, 8), device=d2l.try_gpu())
X[:, 2:6] = 0

Y = corr2d(X, kernel)

conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
conv2d.to(torch.device(d2l.try_gpu()))

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2

for i in range(10000):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 手写梯度下降
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i+1) % 200 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1, 2)))
