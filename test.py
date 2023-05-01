import time
import torch

# 计算量较大的任务
X = torch.rand((10000, 10000))
Y = X.cuda(0)
time_start = time.time()
Z = torch.mm(X, X)
time_end = time.time()
print(f'cpu time cost: {round((time_end - time_start) * 1000, 2)}ms')
time_start = time.time()
Z = torch.mm(Y, Y)
time_end = time.time()
print(f'gpu time cost: {round((time_end - time_start) * 1000, 2)}ms')

# 计算量很小的任务
X = torch.rand((100, 100))
Y = X.cuda(0)
time_start = time.time()
Z = torch.mm(X, X)
time_end = time.time()
print(f'cpu time cost: {round((time_end - time_start) * 1000)}ms')
time_start = time.time()
Z = torch.mm(Y, Y)
time_end = time.time()
print(f'gpu time cost: {round((time_end - time_start) * 1000)}ms')