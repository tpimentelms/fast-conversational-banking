import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class MultiLinear(nn.Module):
    r"""Applies a MultiLinear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        num_linears: number of independent linear layers to use
        bias: If set to False, the layer will not learn an additive bias. Default: True

    Shape:
        - Input: :math:`(N, num_linears, in\_features)`
        - Output: :math:`(N, num_linears, out\_features)`

    Attributes:
        linear: each linear contains:
            weight: the learnable weights of the module of shape (num_linears x out_features x in_features)
            bias:   the learnable bias of the module of shape (num_linears x out_features)

    Examples::

        >>> m = nn.MultiLinear(20, 30, 5)
        >>> input = autograd.Variable(torch.randn(128, 5, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, num_linears, bias=True):
        super(MultiLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_linears = num_linears

        linears = [None] * num_linears
        for i in range(num_linears):
            linears[i] = nn.Linear(in_features, out_features, bias=bias)

        self.linears = ListModule(*linears)

    def forward(self, input):
        assert self.num_linears >= input.size(1), \
            "Error, MultiLinear can receive at most size %d in dimension 1, received %d." \
            % (self.num_linears, input.size(1))
        in_size = min(self.num_linears, input.size(1))

        result = Variable(input.data.new(input.size(0), in_size, self.out_features))
        for i in range(in_size):
            result[:, i, :] = self.linears[i](input[:, i, :])
        return result

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ') ' \
            + '[' + str(self.num_linears) + ']'


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
