import random
import torch

from model.rnn.nn.gru_module import EncoderRNN
from model.rnn.nn.attn_gru_module import AttnDecoderRNN
from model.rnn.network.rnn import RNN


class AttnRNN(RNN):
    def __init__(self, criterion, input_size, output_size, max_length, SOS_token, EOS_token, PAD_token, layers=1, hidden_size=256, dropout=0, teacher_forcing_ratio=1.0):
        encoder = EncoderRNN(input_size, hidden_size, nlayers=layers, dropout=dropout)
        decoder = AttnDecoderRNN(hidden_size, output_size, max_length, nlayers=layers, dropout=dropout)

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
        self.name = 'attn-rnn'

    def _full_forward(self, input_variable, target_variable):
        encoder_outputs, encoder_hidden = self.encoder_forward(
            input_variable)
        loss = self.decoder_forward(
            encoder_hidden, encoder_outputs, target_variable)

        return loss

    def decoder_forward(self, encoder_hidden, encoder_outputs, target_variable):
        bsz = target_variable.size(0)
        decoder_input = self.get_variable(torch.LongTensor([[self.SOS_token]] * bsz))
        decoder_hidden = encoder_hidden
        target_length = target_variable.size(1)
        eos = self.torch.ByteTensor(bsz).zero_()
        loss = 0

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
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

    def _translate(self, input_variable, data_parser, max_len=40, append_eos=True):
        encoder_outputs, encoder_hidden = self.encoder_forward(
            input_variable)
        translation = self.decoder_translate(
            encoder_hidden, encoder_outputs, data_parser, max_len, append_eos=append_eos)

        return translation

    def decoder_translate(self, encoder_hidden, encoder_outputs, data_parser, max_len, append_eos=True):
        bsz = encoder_hidden.size(1)
        decoder_input = self.get_variable(torch.LongTensor([[self.SOS_token]] * bsz))
        decoder_hidden = encoder_hidden
        eos = self.torch.ByteTensor(bsz).zero_()
        # decoder_attentions = torch.zeros(bsz, max_len, max_len)

        # decoded_words = [[]] * bsz
        decoded_words = []

        for di in range(max_len):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # decoder_attentions[:, di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[:, 0]
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

        # return decoded_words, decoder_attentions[:, :di + 1, :]
        return decoded_words
