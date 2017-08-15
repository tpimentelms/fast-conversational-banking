from model.base.enc_dec_network import EncDecNetwork


class FullCNN(EncDecNetwork):
    def __init__(self, encoder, decoder, criterion, SOS_token, EOS_token):
        super(FullCNN, self).__init__(encoder, decoder)
        self.criterion = criterion

        self.EOS_token = EOS_token
        self.SOS_token = SOS_token
        self.PAD_token = None

        self.name = 'full-cnn'

    def full_forward(self, input_variable, target_variable):
        encoder_output = self.encoder_forward(
            input_variable)
        loss = self.decoder_forward(
            encoder_output, target_variable)

        return loss

    def encoder_forward(self, input_variable):
        return self.encoder(input_variable)

    def decoder_forward(self, encoder_output, target_variable):
        decoder_output = self.decoder(encoder_output, target_variable.size(1))
        return self.criterion(decoder_output, target_variable.view(-1))

    def translate(self, input_variable, data_parser, max_len=40, append_eos=True):
        encoder_output = self.encoder_forward(
            input_variable)
        # decoder_output = self.decoder(encoder_output, max_size)
        translation = self.decoder_translate(encoder_output, data_parser, max_len, append_eos=append_eos)

        return translation

    def decoder_translate(self, encoder_output, data_parser, max_len, append_eos=True):
        decoded_words = []

        decoder_output = self.decoder(encoder_output, max_len)
        topv, topi = decoder_output.data.topk(1)

        for i in range(len(topi)):
            ni = topi[i, 0]
            if ni == self.EOS_token or (self.PAD_token is not None and ni == self.PAD_token):
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
            'multilinear': self.decoder.multilinear,
            'max_len': self.decoder.output_len,
            'layers': self.encoder.n_layers,
            'dropout': self.encoder.dropout,
            'kernel_size': self.encoder.kernel_size,
            'ignore_pad': self.criterion.ignore_pad,
        }

        return checkpoint_data
