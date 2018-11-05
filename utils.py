from __future__ import division
import math
import numpy as np
import torch
from torch.autograd import Variable
import json
import logging


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)
    return l


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / \
            torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            #m.bias.data.fill_(0)
            m.bias.data.normal_(0, 0.01)


def normal(x, mu, sigma, gpu_id, gpu=False):
    pi = np.array([math.pi])
    pi = torch.from_numpy(pi).float()
    if gpu:
        with torch.cuda.device(gpu_id):
            pi = Variable(pi).cuda()
    else:
        pi = Variable(pi)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b


class Buffer:

    def __init__(self, max_size=1):
        self.max_size = max_size
        self.current_size = 0
        self.bf = []
        self.pointer = -1
    
    def push(self, x):
        if self.current_size < self.max_size:
            self.bf.append(x)
            self.pointer += 1
            self.current_size = len(self.bf)
        else:
            self.pointer = (self.pointer + 1) % self.max_size
            self.bf[self.pointer] = x
    
    def is_full(self):
        return self.current_size == self.max_size

    def get(self, i):
        i = (self.pointer + 1 + i) % self.current_size
        return self.bf[i]
    
    def get_last(self):
        return self.bf[self.pointer]


if __name__ == "__main__":
    b = Buffer(5)
    for x in range(101):
        b.push(x)
    print(b.get(0), b.is_full())
    print(b.get(-1))

    b = Buffer(2)
    b.push(2)
    print(b.get(0), b.is_full())
    print(b.get(-1))


