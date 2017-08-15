from torch.autograd import Variable

from model.cnn import FullCNN
from model.cnn.nn import RedFaSPEncoder, RedFaSPDecoder, RedMulNetDecoder
# from model.cnn import ReduceCNN


class ReduceCNN(FullCNN):
    def __init__(self, encoder, decoder, criterion, SOS_token, EOS_token, PAD_token):
        super(ReduceCNN, self).__init__(encoder, decoder, criterion, SOS_token, EOS_token)
        self.PAD_token = PAD_token

        self.name = 'reduce-cnn'

    def decoder_forward(self, encoder_output, target_variable):
        decoder_output = self.decoder(encoder_output)

        ext_target = Variable(target_variable.data.new(target_variable.size(0), self.decoder.output_len))
        ext_target[:] = self.PAD_token
        ext_target[:, :target_variable.size(1)] = target_variable[:, :self.decoder.output_len]

        return self.criterion(decoder_output, ext_target.view(-1))

    def decoder_translate(self, encoder_output, data_parser, max_len, append_eos=True):
        decoded_words = []

        decoder_output = self.decoder(encoder_output)
        topv, topi = decoder_output.data.topk(1)

        for i in range(min(max_len, self.decoder.output_len)):
            ni = topi[i, 0]
            if ni == self.EOS_token:
                if append_eos:
                    decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(data_parser.output_dict.index2word[ni])

        return decoded_words


class RedFaSP(ReduceCNN):
    def __init__(self, criterion, input_size, output_size, input_len, output_len, SOS_token, EOS_token, PAD_token, hidden_size=256, stride=2, kernel_size=5, dropout=0.3, n_layer_dec=False, multilinear=True):
        encoder = RedFaSPEncoder(input_size, input_len, hidden_size, PAD_token, stride=stride, kernel_size=kernel_size, dropout=dropout)
        decoder = self.get_decoder(n_layer_dec, output_size, output_len, hidden_size, stride, kernel_size, dropout, multilinear)

        super(RedFaSP, self).__init__(encoder, decoder, criterion, SOS_token, EOS_token, PAD_token)

        if not self.n_layer_dec:
            self.name = 'redfasp'
        elif self.n_layer_dec == 1:
            self.name = 'redmulnet-1'
        elif self.n_layer_dec == 2:
            self.name = 'redmulnet-2'

    def get_decoder(self, n_layer_dec, output_size, output_len, hidden_size, stride, kernel_size, dropout, multilinear):
        self.n_layer_dec = n_layer_dec
        if not self.n_layer_dec:
            return RedFaSPDecoder(
                output_size, output_len, hidden_size, stride=stride, kernel_size=kernel_size, dropout=dropout, multilinear=multilinear)
        else:
            assert multilinear is True, 'Invalid option single-linear for single-reduce-cnn'
            return RedMulNetDecoder(
                hidden_size, output_size, output_len, n_layers=n_layer_dec, kernel_size=kernel_size, dropout=dropout)


    def get_checkpoint_data(self):
        checkpoint_data = super(RedFaSP, self).get_checkpoint_data()
        checkpoint_data['input_len'] = self.encoder.input_len
        checkpoint_data['stride'] = self.encoder.stride
        checkpoint_data['n_layer_dec'] = self.n_layer_dec

        return checkpoint_data

    def get_n_layer_dec(name):
        if name == 'redfasp':
            return False
        elif name == 'redmulnet-1':
            return 1
        elif name == 'redmulnet-2':
            return 2

        raise NotImplementedError('Invalid name for RedFaSP %s' % (name))
