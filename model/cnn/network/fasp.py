from model.cnn.nn import FaSPEncoder, FaSPDecoder
from model.cnn import FullCNN


class FaSP(FullCNN):
    def __init__(self, criterion, input_size, output_size, input_len, output_len, SOS_token, EOS_token, PAD_token,
                 hidden_size=256, kernel_size=5, dilate=2, dropout=0.3, multilinear=True):
        encoder = FaSPEncoder(
            input_size, input_len, hidden_size, PAD_token,
            kernel_size=kernel_size, dilate=dilate, dropout=dropout)
        decoder = FaSPDecoder(
            output_size, output_len, hidden_size,
            kernel_size=kernel_size, dilate=dilate, dropout=dropout, multilinear=multilinear)

        super(FaSP, self).__init__(encoder, decoder, criterion, SOS_token, EOS_token)

        self.PAD_token = PAD_token
        self.name = 'fasp'

    def get_checkpoint_data(self):
        checkpoint_data = super(FaSP, self).get_checkpoint_data()
        checkpoint_data['input_len'] = self.encoder.input_len
        checkpoint_data['dilate'] = self.encoder.dilate

        return checkpoint_data

    def decoder_forward(self, encoder_output, target_variable):
        if self.decoder.output_len < target_variable.size(1):
            target_variable = target_variable[:, :self.decoder.output_len].contiguous()
        return super(FaSP, self).decoder_forward(encoder_output, target_variable)
