import pickle
import torch
from torch import nn

from model.base.criterion import get_criterion
from model.rnn.network.rnn import RNN
from model.rnn.network.attn_rnn import AttnRNN
# from model.rnn import RNN
# from model.rnn import RNN #, AttnGRUNetwork
from model.cnn.nn import FullCNNEncoder, FullCNNDecoder
# from model.enc_dec_network import AttnGRUNetwork
from model.cnn import FullCNN, FaSP, RedFaSP
from model.bytenet import ByteNet
# from model.attn_cnn import AttnCNN, FullAttnCNN
# from model.attn import AttnNet, IndieAttnNet, SimpleIndieAttnNet

from data_handler.language_dict import PAD_token, SOS_token, EOS_token

teacher_forcing_ratio = 0.5


def load_model(load_dir, input_dict, output_dict, load_last=False, eval=False, cuda=False):
    with open(load_dir + '/model.pckl', 'rb') as f:
        checkpoint_data = pickle.load(f)

    model = checkpoint_data['model'] if 'model' in checkpoint_data else None
    hidden_size = checkpoint_data['hidden_size'] if 'hidden_size' in checkpoint_data else None
    input_len = checkpoint_data['input_len'] if 'input_len' in checkpoint_data else None
    max_len = checkpoint_data['max_len'] if 'max_len' in checkpoint_data else None
    multilinear = checkpoint_data['multilinear'] if 'multilinear' in checkpoint_data else True
    layers = checkpoint_data['layers'] if 'layers' in checkpoint_data else None
    dropout = checkpoint_data['dropout'] if 'dropout' in checkpoint_data else None
    kernel_size = checkpoint_data['kernel_size'] if 'kernel_size' in checkpoint_data else None
    ignore_pad = checkpoint_data['ignore_pad'] if 'ignore_pad' in checkpoint_data else None
    dilate = checkpoint_data['dilate'] if 'dilate' in checkpoint_data else None
    stride = checkpoint_data['stride'] if 'stride' in checkpoint_data else None
    attn_heads = checkpoint_data['attn_heads'] if 'attn_heads' in checkpoint_data else None
    # n_layer_dec = checkpoint_data['n_layer_dec'] if 'n_layer_dec' in checkpoint_data else None
    # single_layer_dec = checkpoint_data['single_layer_dec'] if 'single_layer_dec' in checkpoint_data else False

    if eval:
        ignore_pad = True

    net = get_model(input_dict, output_dict, model, hidden_size, max_len, layers, dropout,
                    kernel_size=kernel_size, dilate=dilate, ignore_pad=ignore_pad, multilinear=multilinear,
                    input_len=input_len, stride=stride, attn_heads=attn_heads, cuda=cuda)
    if not load_last:
        net.load_state_dict(torch.load(load_dir + '/best.pyth7', map_location=lambda storage, loc: storage))
    else:
        net.load_state_dict(torch.load(load_dir + '/last.pyth7', map_location=lambda storage, loc: storage))

    return net


def get_model(
        input_dict, output_dict, model, hidden_size, max_len, layers, dropout, kernel_size=None, dilate=None,
        ignore_pad=None, multilinear=True, input_len=None, stride=None, attn_heads=None, cuda=False):
    ignore_pad = ignore_pad if ignore_pad is not None else True
    criterion = get_criterion(output_dict.n_words, PAD_token, ignore_pad=ignore_pad)

    if model == 'rnn' or model == 'attn-rnn':
        attn_rnn = (model == 'attn-rnn')
        net = get_gru(criterion, input_dict, output_dict, hidden_size, max_len, layers, dropout, ignore_pad, attn_rnn)
    elif model == 'full-cnn':
        assert kernel_size is not None, 'kernel_size is necessary for loading %s' % model
        assert ignore_pad is not None, 'ignore_pad is necessary for loading %s' % model
        dilate = dilate if dilate is not None else 2
        net = get_full_cnn(criterion, input_dict, output_dict, hidden_size, max_len, layers, kernel_size, dilate, dropout, ignore_pad, multilinear)
    elif model == 'fasp':
        assert kernel_size is not None, 'kernel_size is necessary for loading %s' % model
        assert dilate is not None, 'dilate is necessary for loading %s' % model
        assert input_len is not None, 'input_len is necessary for loading %s' % model
        net = get_fasp(criterion, input_dict, output_dict, hidden_size, input_len, max_len, kernel_size, dilate, dropout, ignore_pad, multilinear)
    # elif model == 'reduce-cnn':
    #     assert kernel_size is not None, 'kernel_size is necessary for loading reduce cnn network'
    #     assert stride is not None, 'stride is necessary for loading reduce cnn network'
    #     net = get_reduce_cnn(criterion, input_dict, output_dict, hidden_size, input_len, max_len, layers, kernel_size, stride, dropout, ignore_pad, multilinear)
    elif model == 'redfasp' or model == 'redmulnet-1' or model == 'redmulnet-2':
        assert kernel_size is not None, 'kernel_size is necessary for loading full %s' % model
        assert input_len is not None, 'input_len is necessary for loading full %s' % model
        assert stride is not None, 'stride is necessary for loading %s' % model
        net = get_redfasp(criterion, model, input_dict, output_dict, hidden_size, input_len, max_len, kernel_size, stride, dropout, ignore_pad, multilinear)
    # elif model == 'layer-bytenet':
    #     assert kernel_size is not None, 'kernel_size is necessary for loading %s' % model
    #     assert input_len is not None, 'input_len is necessary for loading %s' % model
    #     dilate = dilate if dilate is not None else 2
    #     net = get_layer_bytenet(criterion, input_dict, output_dict, hidden_size, max_len, layers, kernel_size, dilate, dropout, ignore_pad, multilinear, input_len)
    elif model == 'bytenet':
        assert kernel_size is not None, 'kernel_size is necessary for loading %s' % model
        assert input_len is not None, 'input_len is necessary for loading %s' % model
        dilate = dilate if dilate is not None else 2
        net = get_bytenet(criterion, model, input_dict, output_dict, hidden_size, max_len, layers, kernel_size, dilate, dropout, ignore_pad, multilinear, input_len)
    # elif model == 'attn-cnn':
    #     assert kernel_size is not None, 'kernel_size is necessary for loading full reduce cnn network'
    #     assert ignore_pad is not None, 'ignore_pad is necessary for loading full reduce cnn network'
    #     assert input_len is not None, 'input_len is necessary for loading full reduce cnn network'
    #     assert stride is not None, 'stride is necessary for loading full reduce cnn network'
    #     # single_layer_dec = 1 if (model == 'single-reduce-cnn') else False
    #     net = get_attn_cnn(model, input_dict, output_dict, hidden_size, input_len, max_len, kernel_size, stride, dropout, ignore_pad, multilinear)
    # elif model == 'full-attn-cnn':
    #     assert kernel_size is not None, 'kernel_size is necessary for loading full attn cnn network'
    #     assert ignore_pad is not None, 'ignore_pad is necessary for loading full attn cnn network'
    #     assert input_len is not None, 'input_len is necessary for loading full attn cnn network'
    #     assert dilate is not None, 'dilate is necessary for loading full attn cnn network'
    #     # single_layer_dec = 1 if (model == 'single-reduce-cnn') else False
    #     net = get_full_attn_cnn(model, input_dict, output_dict, hidden_size, input_len, max_len, kernel_size, dilate, dropout, ignore_pad, multilinear)
    # elif model == 'attn' or model == 'indie-attn' or model == 'simple-indie-attn':
    #     assert attn_heads is not None, 'attn_heads is necessary for loading attn network'
    #     assert ignore_pad is not None, 'ignore_pad is necessary for loading attn network'
    #     net = get_attn(model, input_dict, output_dict, layers, hidden_size, attn_heads, dropout, ignore_pad)
    else:
        raise NotImplementedError('Non implemented model arquitecture %s' % (model))

    if cuda:
        net.cuda()

    return net


# def get_rnn(input_dict, output_dict, hidden_size, max_len, layers, dropout):
#     encoder = EncoderRNN(input_dict.n_words, hidden_size, nlayers=layers)
#     # decoder = DecoderRNN(hidden_size, output_dict.n_words, nlayers=layers, dropout=dropout)
#     decoder = AttnDecoderRNN(hidden_size, output_dict.n_words, max_len, nlayers=layers)
#     criterion = nn.NLLLoss()
#     net = AttnGRUNetwork(encoder, decoder, criterion, max_len, SOS_token, EOS_token, teacher_forcing_ratio)
#     use_batch = False

#     return net, use_batch


def get_gru(criterion, input_dict, output_dict, hidden_size, max_len, layers, dropout, ignore_pad=True, attn_rnn=False):
    if not attn_rnn:
        net = RNN(
            criterion, input_dict.n_words, output_dict.n_words, max_len,
            SOS_token, EOS_token, PAD_token, hidden_size=hidden_size, layers=layers,
            dropout=dropout, teacher_forcing_ratio=teacher_forcing_ratio)
    else:
        net = AttnRNN(
            criterion, input_dict.n_words, output_dict.n_words, max_len,
            SOS_token, EOS_token, PAD_token, hidden_size=hidden_size, layers=layers,
            dropout=dropout, teacher_forcing_ratio=teacher_forcing_ratio)

    return net


def get_full_cnn(criterion, input_dict, output_dict, hidden_size, max_len, layers, kernel_size, dilate, dropout, ignore_pad, multilinear=True):
    encoder = FullCNNEncoder(input_dict.n_words, hidden_size, n_layers=layers, kernel_size=kernel_size, dilate=dilate, dropout=dropout)
    decoder = FullCNNDecoder(hidden_size, output_dict.n_words, max_len, n_layers=layers, kernel_size=kernel_size, dilate=dilate, dropout=dropout, multilinear=multilinear)
    net = FullCNN(encoder, decoder, criterion, SOS_token, EOS_token)

    return net


def get_fasp(criterion, input_dict, output_dict, hidden_size, input_len, max_len, kernel_size, dilate, dropout, ignore_pad, multilinear=True):
    net = FaSP(
        criterion, input_dict.n_words, output_dict.n_words, input_len, max_len, SOS_token, EOS_token, PAD_token,
        hidden_size=hidden_size, kernel_size=kernel_size, dilate=dilate, dropout=dropout, multilinear=multilinear)

    return net


# def get_reduce_cnn(criterion, input_dict, output_dict, hidden_size, input_len, max_len, layers, kernel_size, stride, dropout, ignore_pad, multilinear=True):
#     # encoder = KMaxEncoderCNN(input_dict.n_words, hidden_size, n_layers=layers, kernel_size=kernel_size, dropout=dropout)
#     encoder = ReduceEncoderCNN(input_dict.n_words, input_len, hidden_size, PAD_token, n_layers=layers, kernel_size=kernel_size, stride=stride, dropout=dropout)
#     decoder = ExpandDecoderCNN(output_dict.n_words, max_len, hidden_size, n_layers=layers, kernel_size=kernel_size, dropout=dropout, multilinear=multilinear)
#     net = ReduceCNN(encoder, decoder, criterion, SOS_token, EOS_token, PAD_token)

#     return net


def get_redfasp(criterion, model, input_dict, output_dict, hidden_size, input_len, max_len, kernel_size, stride, dropout, ignore_pad, multilinear=True):
    n_layer_dec = RedFaSP.get_n_layer_dec(model)
    net = RedFaSP(
        criterion, input_dict.n_words, output_dict.n_words, input_len, max_len,
        SOS_token, EOS_token, PAD_token, hidden_size=hidden_size, kernel_size=kernel_size, stride=stride,
        dropout=dropout, n_layer_dec=n_layer_dec, multilinear=multilinear)

    return net


# def get_layer_bytenet(criterion, input_dict, output_dict, hidden_size, max_len, layers, kernel_size, dilate, dropout, ignore_pad, multilinear=True, input_len=None):
#     net = LayerByteNet(criterion, SOS_token, EOS_token, PAD_token, input_dict.n_words,
#                   output_dict.n_words, max_len, hidden_size, n_layers=layers,
#                   kernel_size=kernel_size, dilate=dilate, dropout=dropout, multilinear=multilinear)

#     return net


def get_bytenet(criterion, model, input_dict, output_dict, hidden_size, max_len, layers, kernel_size, dilate, dropout, ignore_pad, multilinear=True, input_len=None):
    net = ByteNet(criterion, SOS_token, EOS_token, PAD_token, input_dict.n_words,
                  output_dict.n_words, input_len, max_len, hidden_size,
                  kernel_size=kernel_size, dilate=dilate, dropout=dropout, multilinear=multilinear)

    return net


# def get_attn_cnn(model, input_dict, output_dict, hidden_size, input_len, max_len, kernel_size, stride, dropout, ignore_pad, multilinear=True):
#     # n_layer_dec = FullReduceConvNetwork.get_n_layer_dec(model)
#     criterion = get_criterion(output_dict.n_words, PAD_token, ignore_pad=ignore_pad)
#     net = AttnCNN(
#         criterion, input_dict.n_words, output_dict.n_words, input_len, max_len,
#         SOS_token, EOS_token, PAD_token, hidden_size=hidden_size, kernel_size=kernel_size,
#         stride=stride, dropout=dropout, multilinear=multilinear)
#     use_batch = True

#     return net, use_batch


# def get_full_attn_cnn(model, input_dict, output_dict, hidden_size, input_len, max_len, kernel_size, dilate, dropout, ignore_pad, multilinear=True):
#     # n_layer_dec = FullReduceConvNetwork.get_n_layer_dec(model)
#     criterion = get_criterion(output_dict.n_words, PAD_token, ignore_pad=ignore_pad)
#     net = FullAttnCNN(
#         criterion, input_dict.n_words, output_dict.n_words, input_len, max_len,
#         SOS_token, EOS_token, PAD_token, hidden_size=hidden_size, kernel_size=kernel_size,
#         dilate=dilate, dropout=dropout, multilinear=multilinear)
#     use_batch = True

#     return net, use_batch


# def get_attn(model, input_dict, output_dict, layers, hidden_size, attn_heads, dropout, ignore_pad):
#     criterion = get_criterion(output_dict.n_words, PAD_token, ignore_pad=ignore_pad)
#     if model == 'attn':
#         use_model = AttnNet
#     elif model == 'indie-attn':
#         use_model = IndieAttnNet
#     elif model == 'simple-indie-attn':
#         use_model = SimpleIndieAttnNet

#     net = use_model(
#         criterion, input_dict.n_words, output_dict.n_words,
#         SOS_token, EOS_token, PAD_token, layers=layers,
#         hidden_size=hidden_size, attn_heads=attn_heads, dropout=dropout)
#     use_batch = True

#     return net, use_batch
