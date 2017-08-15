from collections import OrderedDict
import torch
from torch import nn
from torch.autograd import Variable

from model.cnn.nn import FullCNNEncoder, FullCNNDecoder
from model.base.multi_linear import MultiLinear


class BaseBytenetEncoder(FullCNNEncoder):
    pass


class BaseBytenetDecoder(FullCNNDecoder):
    def __init__(self, hidden_size, output_size, output_len, n_layers=1, kernel_size=5, dilate=1, dropout=0.3, multilinear=True):
        super(BaseBytenetDecoder, self).__init__(
            hidden_size, output_size, output_len, n_layers=n_layers, 
            kernel_size=kernel_size, dilate=dilate, dropout=dropout,
            multilinear=multilinear)

        self.embedding = nn.Embedding(output_size, hidden_size)

    def forward(self, input, in_tgt):
        outlen = in_tgt.size(1)
        if self.multilinear and outlen > self.output_len:
            outlen = self.output_len
            in_tgt = in_tgt[:, :outlen]

        temp_len = max(outlen, input.size(2))

        ext_input = Variable(input.data.new(input.size(0), input.size(1), temp_len).zero_())
        ext_input[:, :, :input.size(2)] = input[:, :, :]

        in_tgt = self.embedding(in_tgt).transpose(1, 2)
        ext_in_tgt = Variable(input.data.new(in_tgt.size(0), in_tgt.size(1), temp_len).zero_())
        ext_in_tgt[:, :, :in_tgt.size(2)] = in_tgt[:, :, :]

        inner_representations = torch.cat([ext_input, ext_in_tgt], 1)

        representations = self.masked_conv(inner_representations)
        representations = representations[:, :outlen, :].contiguous()

        if not self.multilinear:
            return self.softmax(self.outnet(representations.view(-1, self.hidden_size)))
        else:
            return self.softmax(self.outnet(representations).view(-1, self.output_size))

    def masked_conv(self, input):
        # convolution expands data, remove final collumns which will equal a masked convolution
        outlen = input.size(2)
        representations = self.convnet(input)[:, :, :outlen].transpose(1, 2)
        return representations.contiguous()

    def get_conv_net(self):
        self.modules = OrderedDict()
        self.add_module(2 * self.hidden_size, self.hidden_size, 1)
        for i in range(2, self.n_layers + 1):
            self.add_module(self.hidden_size, self.hidden_size, i)
        return nn.Sequential(self.modules)

    def get_paddings(self):
        return [(self.kernel_size - 1) * self.dilations[i] for i in range(self.n_layers)]
