from util import arg_parser as parser


# Data
parser.add_argument('--save-dir', type=str, default=None,
                    help='Folder where to save model data. None for no checkpoint (default: None)')
parser.add_argument('--save-every', type=int, default=1,
                    help='Save checkpoint every n epochs. (default: 1)')
parser.add_argument('--train-from', type=str, default=None,
                    help='Folder from which to load training data. None for unitialized training (default: None)')
parser.add_argument('--reduce-ratio', type=float, default=1.0,
                    help='If reduced training set is desired, percentage of dataset to use for training (default: 1.0)')
# Model
parser.add_argument('--model', type=str, default='fasp',
                    help='Model architecture. [Options: gru|attn-gru|fasp|redfasp|full-bytenet|single-reduce-cnn|double-reduce-cnn] (default: fasp)')
#                   help='Model architecture. [Options: rnn|gru|attn-gru|cnn|reduce-cnn|full-cnn|full-reduce-cnn|single-reduce-cnn|double-reduce-cnn|bytenet] (default: rnn)')
parser.add_argument('--hidden-size', type=int, default=256,
                    help='Hidden size of rnn/cnn layers (default: 256)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout probability (default: 0.2)')
# # RNN
parser.add_argument('--layers', type=int, default=1,
                    help='Number of layers in model (default: 1)')
# # CNN / ByteNet
parser.add_argument('--kernel-size', type=int, default=3,
                    help='Kernel window size of cnn layers [cnn/reduce-cnn specific] (default: 3)')
parser.add_argument('--dilate', type=int, default=2,
                    help='Dilation value to use in cnn layers (default: 2)')
parser.add_argument('--stride', type=int, default=2,
                    help='Stride value to use in reduce cnn layers (default: 2)')
parser.add_argument('--multi-linear', dest='single_linear', action='store_false',
                    help='Use MultiLinear layer in output of cnn/bytenet layers, instead of common linear (default: False)')
parser.set_defaults(single_linear=True)
# # Attn is all you need
parser.add_argument('--attn-heads', type=int, default=4,
                    help='Number of attn heads in AttnNet (default: 4)')
# Optim
parser.add_argument('--optim', type=str, default='sgd',
                    help='Optim algorithm. [Options: sgd|momentum|adam|rmsprop] (default: sgd)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate (default: 0.01)')
parser.add_argument('--epochs', type=float, default=1,
                    help='Number of epochs (default: 1)')
parser.add_argument('--improve-wait', type=int, default=50,
                    help='Number of epochs to wait for validation improvement before early stopping (default: 50)')
parser.add_argument('--init-params', type=float, default=0.05,
                    help='Initial parameters weight range (default: 0.05)')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay (L2 penalty). (default: 0.0)')
parser.add_argument('--ignore-pad', action='store_false',
                    help='Ignore pad in NLLLoss criterion (default: True)')
parser.set_defaults(ignore_pad=True)
# Others
parser.add_argument('--print-every', type=int, default=50,
                    help='Number of iterations between print msg (default: 50)')


def add_argument(*args, **kwargs):
    return parser.add_argument(*args, **kwargs)


def parse_args():
    return parser.parse_args()
