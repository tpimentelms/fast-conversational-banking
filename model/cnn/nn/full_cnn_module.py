import math
from collections import OrderedDict
import torch.nn as nn
from torch.autograd import Variable

from model.cnn.nn import CNNEncoder, CNNDecoder
from model.base.multi_linear import MultiLinear


class FullCNNBase(object):
    def config_size(self, n_layers, kernel_size, dilate):
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        self.dilate = dilate
        self.dilations = [(self.dilate ** i) for i in range(self.n_layers)]
        self.paddings = self.get_paddings()

    def add_module(self, in_channels, out_channels, layer):
        self.modules['conv%d' % layer] = nn.Conv1d(
            in_channels, out_channels, self.kernel_size, padding=self.paddings[layer - 1], dilation=self.dilations[layer - 1])
        # self.modules['relu%d' % layer] = nn.ReLU()
        self.modules['elu%d' % layer] = nn.ELU()
        # self.modules['batch%d' % layer] = nn.BatchNorm1d(out_channels)
        self.modules['drop%d' % layer] = nn.Dropout(p=self.dropout)

    def get_paddings(self):
        return [math.floor(self.kernel_size / 2) * self.dilations[i] for i in range(self.n_layers)]


class FullCNNEncoder(FullCNNBase, CNNEncoder):
    def __init__(self, input_size, hidden_size, n_layers=1, kernel_size=5, dilate=1, dropout=0.3):
        self.config_size(n_layers, kernel_size, dilate)

        super(FullCNNEncoder, self).__init__(
            input_size, hidden_size, n_layers=n_layers, kernel_size=kernel_size,
            dropout=dropout)


class FullCNNDecoder(FullCNNBase, CNNDecoder):
    def __init__(self, hidden_size, output_size, output_len, n_layers=1, kernel_size=5, dilate=1, dropout=0.3, multilinear=True):
        self.config_size(n_layers, kernel_size, dilate)
        

        super(FullCNNDecoder, self).__init__(
            hidden_size, output_size, output_len, n_layers=n_layers,
            kernel_size=kernel_size, dropout=dropout, multilinear=multilinear)

    def forward(self, input, outlen):
        if self.multilinear and outlen > self.output_len:
            print('Warning: FullCNN decoder should receive at most size %d as outlen, received %d.'
                % (self.output_len, outlen))

            outlen = self.output_len
        temp_len = max(outlen, input.size(2))

        ext_input = Variable(input.data.new(input.size(0), input.size(1), temp_len).zero_())
        ext_input[:, :, :input.size(2)] = input[:, :, :]

        representations = self.convnet(ext_input).transpose(1, 2)
        representations = representations[:, :outlen, :].contiguous()
        if self.multilinear:
            return self.softmax(self.outnet(representations).view(-1, self.output_size))
        else:
            return self.softmax(self.outnet(representations.view(-1, self.hidden_size)))

    def get_context(self, input, outlen):
        if self.multilinear and outlen > self.output_len:
            outlen = self.output_len
        temp_len = max(outlen, input.size(2))

        ext_input = Variable(input.data.new(input.size(0), input.size(1), temp_len).zero_())
        ext_input[:, :, :input.size(2)] = input[:, :, :]

        representations = self.convnet(ext_input).transpose(1, 2)
        representations = representations[:, :outlen, :].contiguous()
        if self.multilinear:
            return self.outnet(representations)
        else:
            batch_size = representations.size(0)
            output_len = representations.size(1)
            return self.outnet(representations.view(-1, self.hidden_size)) \
                       .view(batch_size, output_len, self.output_size)
