import torch
x = torch.tensor([2,0])
y = torch.tensor([[1,9,2],[4,3,6],[7,5,8]])
print(y[[2,0],[0,2]])
print(y[[0,2,2],[2,0,1]])
print(len(y))
print(y.argmax(axis=0))
print(y.argmax(axis=1))
print([0.]*3)