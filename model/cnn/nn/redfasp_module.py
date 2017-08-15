import math
import torch.nn as nn
import torch.nn.functional as F

from model.cnn.nn import CNNEncoder, CNNDecoder
from model.base.multi_linear import MultiLinear


class RedFaSPBase(object):
    def config_size(self, length, stride, kernel_size):
        n_layers = self.get_n_layers(length, stride, kernel_size)
        self.stride = stride

        return n_layers

    def get_paddings(self):
        return self.paddings

    def add_module(self, in_channels, out_channels, layer):
        self.modules['conv%d' % layer] = self.conv_module(
            in_channels, out_channels, self.kernel_size, padding=self.paddings[layer - 1], stride=self.stride)
        # self.modules['relu%d' % layer] = nn.ReLU()
        self.modules['elu%d' % layer] = nn.ELU()
        # self.modules['batch%d' % layer] = nn.BatchNorm1d(out_channels)
        self.modules['drop%d' % layer] = nn.Dropout(p=self.dropout)



class RedFaSPEncoder(RedFaSPBase, CNNEncoder):
    def __init__(self, input_size, input_len, hidden_size, PAD_token, n_layers=1, stride=2, kernel_size=5, dropout=0.3):
        # n_layers = self.get_n_layers(input_len, stride, kernel_size)
        n_layers = self.config_size(input_len, stride, kernel_size)
        self.conv_module = nn.Conv1d
        self.input_len = input_len
        self.PAD_token = PAD_token
        # self.stride = stride

        super(RedFaSPEncoder, self).__init__(
            input_size, hidden_size, n_layers=n_layers, kernel_size=kernel_size,
            dropout=dropout)

    def forward(self, input):
        ext_input = self.extend_var(input, self.input_len, self.PAD_token)

        embedded = self.embedding(ext_input)
        output = self.convnet(embedded.transpose(1, 2))

        return output

    def get_n_layers(self, input_len, stride=None, kernel_size=None):
        stride = stride if stride is not None else self.stride
        kernel_size = kernel_size if kernel_size is not None else self.kernel_size

        layers = 0
        output_len = math.ceil((input_len - kernel_size + 1) * 1.0 / stride)
        while output_len > 0:
            input_len = output_len
            output_len = math.ceil((input_len - kernel_size + 1) * 1.0 / stride)
            layers += 1

        self.paddings = [0] * layers
        if input_len > 1:
            self.paddings += [math.ceil((kernel_size - input_len) / 2.0)]
            layers += 1

        return layers



class RedFaSPDecoder(RedFaSPBase, CNNDecoder):
    def __init__(self, output_size, output_len, hidden_size, n_layers=1, stride=2, kernel_size=5, dropout=0.3, multilinear=True):
        n_layers = self.config_size(output_len, stride, kernel_size)
        self.conv_module = nn.ConvTranspose1d
        # n_layers = self.get_n_layers(output_len, stride, kernel_size)
        # self.stride = stride

        super(RedFaSPDecoder, self).__init__(
            hidden_size, output_size, output_len, n_layers=n_layers,
            kernel_size=kernel_size, dropout=dropout, multilinear=multilinear)

    def forward(self, input):
        representations = self.deconv(input).transpose(1, 2).contiguous()
        if self.multilinear:
            return self.softmax(self.outnet(representations).view(-1, self.output_size))
        else:
            return self.softmax(self.outnet(representations.view(-1, self.hidden_size)))

    def get_context(self, input):
        representations = self.deconv(input).transpose(1, 2).contiguous()
        if self.multilinear:
            return self.outnet(representations)
        else:
            return representations

    def deconv(self, input):
        output = input
        for i in range(1, self.n_layers + 1):
            # import ipdb; ipdb.set_trace()
            output = self.modules['conv%d' % i](output, output_size=(self.sizes[-i - 1],))
            output = self.modules['elu%d' % i](output)
            output = self.modules['drop%d' % i](output)
        return output

    def get_n_layers(self, input_len, stride=None, kernel_size=None):
        stride = stride if stride is not None else self.stride
        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        self.sizes = [input_len]

        layers = 0
        output_len = math.ceil((input_len - kernel_size + 1) * 1.0 / stride)
        self.sizes += [output_len]

        while output_len > 0:
            input_len = output_len
            output_len = math.ceil((input_len - kernel_size + 1) * 1.0 / stride)
            self.sizes += [output_len]
            layers += 1

        self.sizes = self.sizes[:-1]
        self.paddings = [0] * layers
        if input_len > 1:
            self.sizes += [1]
            self.paddings = [math.ceil((kernel_size - input_len) / 2.0)] + self.paddings
            layers += 1

        return layers
