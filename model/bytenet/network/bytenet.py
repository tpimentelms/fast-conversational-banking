from model.bytenet.nn import BytenetEncoder, BytenetDecoder
from model.bytenet import BaseByteNet


class ByteNet(BaseByteNet):
    def __init__(
            self, criterion, SOS_token, EOS_token, PAD_token, input_size, output_size, input_len,
            output_len, hidden_size, kernel_size=5, dilate=2, dropout=0.3, multilinear=True):
        encoder = BytenetEncoder(
            input_size, input_len, hidden_size, PAD_token, kernel_size=kernel_size,
            dilate=dilate, dropout=dropout)
        decoder = BytenetDecoder(
            hidden_size, output_size, output_len, kernel_size=kernel_size,
            dilate=dilate, dropout=dropout, multilinear=multilinear)

        super(BaseByteNet, self).__init__(encoder, decoder, criterion, SOS_token, EOS_token)

        self.PAD_token = PAD_token
        self.max_len = output_len
        self.input_len = input_len

        self.name = 'bytenet'

    def get_checkpoint_data(self):
        checkpoint_data = super(ByteNet, self).get_checkpoint_data()
        checkpoint_data['input_len'] = self.input_len

        return checkpoint_data
