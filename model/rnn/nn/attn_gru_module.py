import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, nlayers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nlayers = nlayers
        self.dropout_p = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, nlayers, dropout=dropout)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        bsz = input.size(0)
        embedded = self.dropout(self.embedding(input).view(bsz, -1))

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        output2 = torch.cat((embedded, attn_applied), 1)
        attn_input = F.relu(self.attn_combine(output2)).unsqueeze(0)

        representations, hidden = self.gru(attn_input, hidden)
        representations = self.dropout(representations)

        output = F.log_softmax(self.out(representations[0]))
        return output, hidden, attn_weights

    def initHidden(self, bsz=1):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.hidden_size).zero_())
