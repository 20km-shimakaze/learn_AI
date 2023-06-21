import torch
import math
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

batch_size, num_steps = 32, 35
# vocab 根据index转成对应的词
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# 时间步数， 批量大小， 词表大小
# 独热编码
X = torch.arange(10).reshape(2, 5)


# print(F.one_hot(torch.arange(10).reshape(2, 5).T, 28))


def get_params(vocab_size, num_hiddens, device):
    # vocab大小, 隐藏层大小, 设备类型
    """返回可以学习的参数"""
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device))


def rnn(inputs, state, params):
    # 所有的时间步(x0~xt) 初始化的隐藏状态 可学习的参数
    """在一个时间步内计算隐藏状态和输出"""
    W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    for X in inputs:
        # 先拿时间0的批量大小 x vocab_size
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        # tanh 激活函数
        Y = H @ W_hq + b_q
        outputs.append(Y)
    # 行：批量大小*时间长度 列:vocab_size
    return torch.cat(outputs, dim=0), (H)


class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
print(Y.shape, len(new_state), new_state[0].shape)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新的字符"""
    # 生成初始的隐藏状态
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_inputs = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 先将prefix的数据初始化state(预热)
    for y in prefix[1:]:
        _, state = net(get_inputs(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_inputs(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))

def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练模型一个迭代周期"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=)
