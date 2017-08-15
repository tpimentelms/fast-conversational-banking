import argparse
import random

from .logistic_parser import LogisticDataParser as DataParser
from .handler import get_eval_data, write_dataset

parser = argparse.ArgumentParser(description='Pimentel NMT')
# Language
parser.add_argument('--lang1', type=str, default='en',
                    help='language 1 (default: en)')
parser.add_argument('--lang2', type=str, default='fra',
                    help='language 2 (default: fra)')
# Data Files
parser.add_argument('--train-src', type=str, default='data/en-fra/train.en.txt',
                    help='Train source language file (default: data/en-fra/train.en.txt)')
parser.add_argument('--train-tgt', type=str, default='data/en-fra/train.fr.txt',
                    help='Train target language file (default: data/en-fra/train.fr.txt)')
parser.add_argument('--val-src', type=str, default=None,
                    help='Validation source file path. None to use part of training. (default: None)')
parser.add_argument('--val-tgt', type=str, default=None,
                    help='Validation target file path. None to use part of training. (default: None)')
parser.add_argument('--test-src', type=str, default=None,
                    help='Test source file path. None to use part of training. (default: None)')
parser.add_argument('--test-tgt', type=str, default=None,
                    help='Test target file path. None to use part of training. (default: None)')
parser.add_argument('--output-file', type=str, default='data/en-fra/data.pckl',
                    help='Input file (default: data/en-fra/data.pckl)')
# Dataset Args
parser.add_argument('--max-len', type=int, default=100,
                    help='Max considered length of input and output sentences, rest is ignored (default: 40)')
parser.add_argument('--min-count', type=int, default=1,
                    help='Min count of word to be in input dict. (default: 1)')
parser.add_argument('--remove-brackets', action='store_true',
                    help='Remove parenthesis and commas from dataset (default: False)')
# Options
parser.add_argument('--quiet', action='store_true',
                    help='Print no output while creating dataset (default: False)')
# Others
parser.add_argument('--seed', type=int, default=777,
                    help='Seed for random algorithms repeatability (default: 777)')
args = parser.parse_args()
random.seed(args.seed)


def main(args):
    data_parser = DataParser(args.max_len, cuda=True, quiet=args.quiet, remove_brackets=args.remove_brackets)
    train_data = data_parser.read_data(args.train_src, args.train_tgt, max_len=args.max_len)
    random.shuffle(train_data)

    val_size = int(0.1 * len(train_data))

    train_data, test_data = get_eval_data(data_parser, args.test_src, args.test_tgt, train_data, val_size)
    train_data, val_data = get_eval_data(data_parser, args.val_src, args.val_tgt, train_data, val_size)

    data_parser.setup_parser(train_data)
    data_parser.remove_rare_words(args.min_count)

    write_dataset(train_data, val_data, test_data, data_parser, args.output_file)


if __name__ == '__main__':
    main(args)
