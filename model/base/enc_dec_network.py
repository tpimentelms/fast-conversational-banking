import random
import pickle
import torch
from torch import nn


class EncDecNetwork(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncDecNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self._cuda = False

    def full_forward(self):
        raise NotImplementedError

    def translate(self):
        raise NotImplementedError

    def cuda(self):
        super(EncDecNetwork, self).cuda()
        self.encoder.cuda()
        self.decoder.cuda()

        self._cuda = True

    def initialize_params(self, init_range):
        for p in self.parameters():
            p.data.uniform_(-init_range, init_range)

    def save_config_data(self, path):
        checkpoint_data = self.get_checkpoint_data()
        with open(path, 'wb') as f:
            pickle.dump(checkpoint_data, f, -1)

    def get_checkpoint_data(self):
        raise NotImplementedError('get_checkpoint_data should be implemented by class that inherits EncDecNetwork')
