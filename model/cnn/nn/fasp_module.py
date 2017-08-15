# import math

from model.cnn.nn import FullCNNEncoder, FullCNNDecoder


class BaseFaSP(object):
    def get_n_layers(self, input_len, kernel_size=None, dilate=None):
        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        dilate = dilate if dilate is not None else self.dilate

        layers = 0
        reach = 0
        # while reach < math.ceil(input_len / 2):
        while reach < input_len:
            layers += 1
            reach += (dilate ** (layers)) * ((kernel_size - 1) / 2)

        return layers


class FaSPEncoder(FullCNNEncoder, BaseFaSP):
    def __init__(self, input_size, input_len, hidden_size, PAD_token, kernel_size=3, dilate=2, dropout=0.3):
        n_layers = self.get_n_layers(input_len, kernel_size=kernel_size, dilate=dilate)
        # print(n_layers, input_len)

        super(FaSPEncoder, self).__init__(
            input_size, hidden_size, n_layers=n_layers, kernel_size=kernel_size,
            dilate=dilate, dropout=dropout)

        self.input_len = input_len
        self.PAD_token = PAD_token

    def forward(self, input):
        ext_input = self.extend_var(input, self.input_len, self.PAD_token)

        embedded = self.embedding(ext_input)
        output = self.convnet(embedded.transpose(1, 2))

        return output


class FaSPDecoder(FullCNNDecoder, BaseFaSP):
    def __init__(self, output_size, output_len, hidden_size, kernel_size=3, dilate=2, dropout=0.3, multilinear=True):
        n_layers = self.get_n_layers(output_len, kernel_size=kernel_size, dilate=dilate)

        super(FaSPDecoder, self).__init__(
            hidden_size, output_size, output_len, n_layers=n_layers, kernel_size=kernel_size,
            dilate=dilate, dropout=dropout, multilinear=multilinear)
