import torch
from torch import nn


def corr2d(X, K):
    """矩阵×卷积核"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1), device=X.device)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def corr2d_multi_in(X, K):
    """求所有通道维数卷积后相加"""
    return sum(corr2d(x, k) for x, k in zip(X, K))


def try_gpu(i=0):
    """尝试创建gpu(i)"""
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """尝试返回所有可用的gpu"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]