import math
from collections import OrderedDict
import torch.nn as nn
from torch.autograd import Variable

from model.base.multi_linear import MultiLinear


class CNNBase(nn.Module):
    def __init__(self, hidden_size, n_layers=1, kernel_size=5, dropout=0.3):
        super(CNNBase, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.kernel_size = kernel_size
        self.dropout = dropout

    def get_conv_net(self):
        self.modules = OrderedDict()
        self.add_module(self.hidden_size, self.hidden_size, 1)
        for i in range(2, self.n_layers + 1):
            self.add_module(self.hidden_size, self.hidden_size, i)
        return nn.Sequential(self.modules)

    def add_module(self, in_channels, out_channels, layer):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError

    def extend_var(self, var, ext_len, fill_value=0):
        batch_size = var.size(0)

        ext_input = Variable(var.data.new(batch_size, ext_len).fill_(fill_value))
        ext_input[:, :var.size(1)] = var[:, :ext_len]

        return ext_input

    def get_paddings(self):
        raise NotImplementedError


class CNNEncoder(CNNBase):
    def __init__(self, input_size, hidden_size, n_layers=1, kernel_size=5, dropout=0.3):
        super(CNNEncoder, self).__init__(
            hidden_size, n_layers=n_layers, kernel_size=kernel_size,
            dropout=dropout)

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.convnet = self.get_conv_net()

    def forward(self, input):
        # embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.embedding(input)
        output = self.convnet(embedded.transpose(1, 2))
        return output


class CNNDecoder(CNNBase):
    def __init__(self, hidden_size, output_size, output_len, n_layers=1, kernel_size=5, dropout=0.3, multilinear=True):
        super(CNNDecoder, self).__init__(hidden_size, n_layers=n_layers, kernel_size=kernel_size, dropout=dropout)
        # self.config_size(dilate)
        self.output_size = output_size
        self.output_len = output_len
        self.multilinear = multilinear

        self.convnet = self.get_conv_net()
        if self.multilinear:
            self.outnet = MultiLinear(hidden_size, output_size, self.output_len)
        else:
            self.outnet = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
