from model.bytenet.nn import BaseBytenetDecoder
from model.cnn.nn import FaSPEncoder
from model.cnn.nn.fasp_module import BaseFaSP


class BytenetEncoder(FaSPEncoder):
    pass


class BytenetDecoder(BaseBytenetDecoder, BaseFaSP):
    def __init__(self, hidden_size, output_size, output_len, kernel_size=5, dilate=2, dropout=0.3, multilinear=True):
        # n_layers=self.get_n_layers()
        n_layers = self.get_n_layers(output_len, kernel_size=kernel_size, dilate=dilate)
        self.output_len = output_len

        # super(FullDecoderBytenet, self).__init__(hidden_size, n_layers=n_layers, kernel_size=kernel_size, dilate=dilate, dropout=dropout)
        super(BytenetDecoder, self).__init__(
            hidden_size, output_size, output_len, n_layers=n_layers, kernel_size=kernel_size,
            dilate=dilate, dropout=dropout, multilinear=multilinear)
