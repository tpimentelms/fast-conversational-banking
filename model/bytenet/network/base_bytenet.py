from torch.autograd import Variable

from model.bytenet.nn import BaseBytenetEncoder, BaseBytenetDecoder
from model.cnn import FullCNN


class BaseByteNet(FullCNN):
    def __init__(
            self, criterion, SOS_token, EOS_token, PAD_token, input_size, output_size,
            max_len, hidden_size, n_layers=1, kernel_size=5, dilate=2, dropout=0.3, multilinear=True):
        encoder = EncoderBytenet(
            input_size, hidden_size, n_layers=n_layers, kernel_size=kernel_size,
            dilate=dilate, dropout=dropout)
        decoder = DecoderBytenet(
            hidden_size, output_size, max_len, n_layers=n_layers, kernel_size=kernel_size,
            dilate=dilate, dropout=dropout, multilinear=multilinear)

        super(BaseByteNet, self).__init__(encoder, decoder, criterion, SOS_token, EOS_token)

        self.PAD_token = PAD_token
        self.max_len = max_len

        self.name = 'base-bytenet'

    def decoder_forward(self, encoder_output, target_variable):
        if self.decoder.multilinear and target_variable.size(1) > self.decoder.output_len:
            target_variable = target_variable[:, :self.decoder.output_len].contiguous()

        in_tgt = Variable(target_variable.data.new(target_variable.size(0), target_variable.size(1)))
        in_tgt[:, 0] = self.SOS_token
        in_tgt[:, 1:] = target_variable[:, :-1]

        decoder_output = self.decoder(encoder_output, in_tgt)
        return self.criterion(decoder_output, target_variable.view(-1))

    def decoder_translate(self, encoder_output, data_parser, max_len, append_eos=True):
        if self.decoder.multilinear:
            max_len = min(max_len, self.decoder.output_len)

        decoded_words = []

        decoded_tgts = Variable(encoder_output.data.new(encoder_output.size(0), max_len)).long()
        decoded_tgts[:, 0] = self.SOS_token

        # import ipdb; ipdb.set_trace()

        for i in range(1, max_len + 1):
            in_tgt = Variable(encoder_output.data.new(encoder_output.size(0), i)).long()
            in_tgt.data = decoded_tgts.data[:, :i]

            decoder_output = self.decoder(encoder_output, in_tgt)
            topv, topi = decoder_output.data.topk(1)

            ni = topi[i - 1, 0]
            if i < max_len:
                decoded_tgts[:, i] = ni

            if ni == self.EOS_token or ni == self.PAD_token:
                if append_eos:
                    decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(data_parser.output_dict.index2word[ni])

        return decoded_words

    def get_checkpoint_data(self):
        checkpoint_data = {
            'model': self.name,
            'hidden_size': self.encoder.hidden_size,
            'dilate': self.encoder.dilate,
            'max_len': self.max_len,
            'layers': self.encoder.n_layers,
            'dropout': self.encoder.dropout,
            'kernel_size': self.encoder.kernel_size,
            'ignore_pad': self.criterion.ignore_pad,
            'multilinear': self.decoder.multilinear,
        }

        return checkpoint_data
