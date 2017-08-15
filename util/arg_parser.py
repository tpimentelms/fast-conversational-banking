import argparse

parser = argparse.ArgumentParser(description='Pimentel ConversationalBanking')
# Data
parser.add_argument('--data', type=str, default='data/samantha/dataset.pckl',
                    help='Dataset file (Created with lang.create_dataset.py). (default: data/samantha/dataset.pckl)')
parser.add_argument('--max-len', type=int, default=100,
                    help='Max considered length of input and output sentences, rest is ignored (default: 100)')
# Optim
parser.add_argument('--batch-size', type=int, default=16,
                    help='Size of each batch (default: 16)')
# Others
parser.add_argument('--seed', type=int, default=777,
                    help='Seed for random algorithms repeatability (default: 777)')
parser.add_argument('--cpu', dest='cuda', action='store_false',
                    help='Use cpu instead of gpu (default: use gpu)')
parser.set_defaults(cuda=True)


def add_argument(*args, **kwargs):
    return parser.add_argument(*args, **kwargs)


def set_defaults(*args, **kwargs):
    return parser.set_defaults(*args, **kwargs)


def parse_args():
    return parser.parse_args()
