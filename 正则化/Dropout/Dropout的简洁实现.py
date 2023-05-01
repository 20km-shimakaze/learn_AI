import torch
from torch import nn
from d2l import torch as d2l


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.1, 0.5
net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_hiddens1), nn.ReLU(), nn.Dropout(dropout1),
                    nn.Linear(num_hiddens1, num_hiddens2), nn.ReLU(), nn.Dropout(dropout1),
                    nn.Linear(num_hiddens2, num_outputs))
num_epoch, lr, batch_size = 6, 0.1, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch, trainer)
d2l.plt.show()
