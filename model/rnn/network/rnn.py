import random
import torch
from torch import autograd

from model.base.enc_dec_network import EncDecNetwork
from model.rnn.nn.gru_module import EncoderRNN, DecoderRNN
# from util.util import get_variable


class RNN(EncDecNetwork):
    def __init__(self, criterion, input_size, output_size, max_length, SOS_token, EOS_token, PAD_token, layers=1, hidden_size=256, dropout=0, teacher_forcing_ratio=1.0):
        encoder = EncoderRNN(input_size, hidden_size, nlayers=layers, dropout=dropout)
        decoder = DecoderRNN(hidden_size, output_size, nlayers=layers, dropout=dropout)

        super(RNN, self).__init__(encoder, decoder)
        self.criterion = criterion

        self.nlayers = layers
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.EOS_token = EOS_token
        self.SOS_token = SOS_token
        self.PAD_token = PAD_token
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.torch = torch
        self.name = 'rnn'

    def full_forward(self, input_variable, target_variable):
        if input_variable.is_cuda:
            self.torch = torch.cuda

        return self._full_forward(input_variable, target_variable)

    def _full_forward(self, input_variable, target_variable):
        encoder_outputs, encoder_hidden = self.encoder_forward(
            input_variable)
        loss = self.decoder_forward(
            encoder_hidden, target_variable)

        return loss

    def get_variable(self, input_tensor):
        variable = autograd.Variable(input_tensor)
        return variable.cuda() if self._cuda else variable

    def encoder_forward(self, input_variable):
        bsz = input_variable.size(0)
        encoder_hidden = self.encoder.initHidden(bsz)
        input_length = input_variable.size(1)
        encoder_outputs = self.get_variable(torch.zeros(bsz, self.max_length, self.encoder.hidden_size))
        eos = self.torch.ByteTensor(bsz).zero_()

        temp_hidden = encoder_hidden
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[:, ei], encoder_hidden)

            replace_mask = eos.view(1, bsz, 1).repeat(self.nlayers, 1, self.hidden_size)
            encoder_hidden[replace_mask] = temp_hidden[replace_mask]
            temp_hidden = encoder_hidden
            encoder_outputs[:, ei] = encoder_output[0, :, :]

            eos[input_variable.data[:, ei] == self.EOS_token] = True

        return encoder_outputs, encoder_hidden

    def decoder_forward(self, encoder_hidden, target_variable):
        bsz = target_variable.size(0)
        decoder_input = self.get_variable(torch.LongTensor([[self.SOS_token]] * bsz))
        decoder_hidden = encoder_hidden
        target_length = target_variable.size(1)
        eos = self.torch.ByteTensor(bsz).zero_()
        loss = 0

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            loss += self.criterion(decoder_output, target_variable[:, di])

            if not use_teacher_forcing:
                topv, topi = decoder_output.data.topk(1)
                ni = topi[:, 0]
                ni[eos] = self.PAD_token
                decoder_input = self.get_variable(ni)
                eos[ni == self.EOS_token] = True
                if eos.all():
                    break
            else:
                decoder_input = target_variable[:, di]  # Teacher forcing

        return loss

    def translate(self, input_variable, data_parser, max_len=40, append_eos=True):
        if input_variable.is_cuda:
            self.torch = torch.cuda

        return self._translate(input_variable, data_parser, max_len=max_len, append_eos=append_eos)

    def _translate(self, input_variable, data_parser, max_len=40, append_eos=True):
        encoder_outputs, encoder_hidden = self.encoder_forward(
            input_variable)
        translation = self.decoder_translate(
            encoder_hidden, data_parser, max_len, append_eos=append_eos)

        return translation

    def decoder_translate(self, encoder_hidden, data_parser, max_len, append_eos=True):
        bsz = encoder_hidden.size(1)
        decoder_input = self.get_variable(torch.LongTensor([[self.SOS_token]] * bsz))
        decoder_hidden = encoder_hidden
        eos = self.torch.ByteTensor(bsz).zero_()

        # decoded_words = [[]] * bsz
        decoded_words = []

        for di in range(max_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[:, 0]
            # ni = topi[0][0]
            eos[ni == self.EOS_token] = True

            if eos.all():
                if append_eos:
                    # for i, word in enumerate(ni):
                        # decoded_words[i].append('<EOS>')
                    decoded_words.append('<EOS>')
                break
            else:
                # for i, word in enumerate(ni):
                    # decoded_words[i].append(data_parser.output_dict.index2word[word])
                decoded_words.append(data_parser.output_dict.index2word[ni[0]])
            decoder_input = self.get_variable(ni)

        return decoded_words

    def get_checkpoint_data(self):
        checkpoint_data = {
            'model': self.name,
            'hidden_size': self.encoder.hidden_size,
            'max_len': self.max_length,
            'layers': self.encoder.nlayers,
            'dropout': self.decoder.dropout_p,
            'ignore_pad': self.criterion.ignore_pad,
        }

        return checkpoint_data
