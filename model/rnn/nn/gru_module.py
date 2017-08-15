import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.dropout_p = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, nlayers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self._cuda = False

    def forward(self, input, hidden):
        bsz = input.size(0)
        embedded = self.dropout(self.embedding(input).view(1, bsz, -1))
        output, hidden = self.gru(embedded, hidden)
        return self.dropout(output), hidden

    def initHidden(self, bsz=1):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.hidden_size).zero_())

    def cuda(self):
        super(EncoderRNN, self).cuda()
        self._cuda = True

    def cpu(self):
        super(EncoderRNN, self).cpu()
        self._cuda = False


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, nlayers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.dropout_p = dropout

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, nlayers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        bsz = input.size(0)
        embedded = self.dropout(self.embedding(input).view(1, bsz, -1))
        representations, hidden = self.gru(embedded, hidden)
        representations = self.dropout(representations)
        output = F.log_softmax(self.out(representations[0]))
        # import ipdb; ipdb.set_trace()
        return output, hidden

    def initHidden(self, bsz=1):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.hidden_size).zero_())
