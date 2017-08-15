from util import arg_parser as parser


# Model
parser.add_argument('--load-model', type=str, default=None,
                    help='Folder from which to load model data. (Necessary, no default)')
parser.add_argument('--load-last', action='store_true',
                    help='Load last model. If not set will load best model (default: False)')
# Others
parser.add_argument('--print-errors', action='store_true',
                    help='Save errors model makes. (default: False)')
parser.add_argument('--eval-time', action='store_true',
                    help='Evaluate time model takes to run. (default: False)')


def add_argument(*args, **kwargs):
    return parser.add_argument(*args, **kwargs)


def parse_args():
    return parser.parse_args()
