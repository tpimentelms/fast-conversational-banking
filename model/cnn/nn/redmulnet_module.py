import torch.nn as nn
import torch.nn.functional as F

from model.base.multi_linear import MultiLinear


class RedMulNetDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, output_len, n_layers=1, kernel_size=5, dropout=0.3):
        super(RedMulNetDecoder, self).__init__()
        self.output_size = output_size
        self.output_len = output_len
        self.dropout_p = dropout
        self.n_layers = n_layers
        self.multilinear = True

        if n_layers == 2:
            self.affine1 = MultiLinear(hidden_size, hidden_size, self.output_len)
            self.dropout = nn.Dropout(dropout)
        self.outnet = MultiLinear(hidden_size, output_size, self.output_len)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        representations = input.repeat(1, 1, self.output_len).transpose(1, 2).contiguous()
        if self.n_layers == 2:
            representations = F.relu(self.dropout(self.affine1(representations)))

        return self.softmax(self.outnet(representations).view(-1, self.output_size))

    def get_context(self, input):
        representations = input.repeat(1, 1, self.output_len).transpose(1, 2).contiguous()
        if self.n_layers == 2:
            representations = F.relu(self.dropout(self.affine1(representations)))

        return self.outnet(representations)