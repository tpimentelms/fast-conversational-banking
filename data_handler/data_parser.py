import re
import torch
from torch.autograd import Variable

from .language_dict import LanguageDict, EOS_token, PAD_token


class DataParser(object):
    def __init__(self, max_len, cuda=True, quiet=True):
        self.max_len = max_len

        self.input_max_len = 0
        self.output_max_len = 0

        self.quiet = quiet
        self._cuda = cuda

    # Lowercase, trim, and remove non-letter characters
    def normalize_string(self, s):
        raise NotImplementedError(
            'DataParser class should not be used directly, and ' +
            'class which inherits it should implement normalize_string')

    def read_input(self, src_file, tgt_file):
        self._print("Reading lines...")

        # Read the file and split into lines
        lines1 = open(src_file).read().strip().split('\n')
        lines2 = open(tgt_file).read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[x for x in l] for l in zip(lines1, lines2)]

        return pairs

    def split_sentence(self, p):
        return p.split(' ')

    def parse_pair(self, p, max_len=None):
        sentences = (self.split_sentence(x) for x in p)
        sentences = tuple([self.normalize_string(x) for x in sentences])

        if max_len is not None:
            sentences = [x[:max_len - 1] for x in sentences]

        return sentences

    def parse_pairs(self, pairs, max_len=None):
        return [self.parse_pair(pair, max_len=max_len) for pair in pairs]

    def read_data(self, src_file, tgt_file, max_len=None):
        pairs = self.read_input(src_file, tgt_file)
        self._print("Read %s sentence pairs" % len(pairs))
        pairs = self.parse_pairs(pairs, max_len)
        # import ipdb; ipdb.set_trace()
        return pairs

    def setup_parser(self, pairs):
        # Make dicts instances
        self.input_dict = LanguageDict('src')
        self.output_dict = LanguageDict('tgt')
        self.pairs = pairs

        self._print("Counting words...")
        for pair in self.pairs:
            self.add_src_sentence(pair[0])
            self.add_tgt_sentence(pair[1])
        self.max_len = min(self.max_len, self.output_max_len)

        self._print("Counted words:")
        self._print('\t', self.input_dict.name, self.input_dict.n_words)
        self._print('\t', self.output_dict.name, self.output_dict.n_words)

        return self.input_dict, self.output_dict

    def add_src_sentence(self, sentence):
        self.input_dict.addSentence(sentence)
        self.input_max_len = max(self.input_max_len, len(sentence) + 1)

    def add_tgt_sentence(self, sentence):
        self.output_dict.addSentence(sentence)
        self.output_max_len = max(self.output_max_len, len(sentence) + 1)

    def remove_rare_words(self, min_count):
        self.input_dict.removeRareWords(min_count)
        self._print("\t after reduce", self.input_dict.name, len(self.input_dict.index2word))

    def indexes_from_sentence(lang_dict, sentence):
        return [lang_dict.getWordIndex(word) for word in sentence]

    def variable_from_sentence(self, lang_dict, sentence):
        indexes = DataParser.indexes_from_sentence(lang_dict, sentence)
        indexes.append(EOS_token)
        if self._cuda:
            return Variable(torch.cuda.LongTensor(indexes).view(-1, 1))
        else:
            return Variable(torch.LongTensor(indexes).view(-1, 1))

    # def variables_from_pair(self, pair=None):
    #     pair = self.pair if pair is None else pair
    #     input_variable = self.variable_from_sentence(self.input_dict, pair[0])
    #     target_variable = self.variable_from_sentence(self.output_dict, pair[1])
    #     return input_variable, target_variable

    def variables_from_pairs(self, pairs):
        input_variables = []
        target_variables = []

        for pair in pairs:
            input_variables += [self.variable_from_sentence(self.input_dict, pair[0]).transpose(0, 1)]
            target_variables += [self.variable_from_sentence(self.output_dict, pair[1]).transpose(0, 1)]
        target_avg_len = sum([x.size(1) for x in target_variables]) / len(target_variables)

        input_variable = self.pad_and_cat(input_variables)
        target_variable = self.pad_and_cat(target_variables)

        return input_variable, target_variable, target_avg_len

    def pad_and_cat(self, tensor_list):
        max_len = max([x.size(1) for x in tensor_list])
        pad_list = Variable(tensor_list[0].data.new(len(tensor_list), max_len))
        pad_list[:] = PAD_token

        for i, tensor in enumerate(tensor_list):
            pad_list[i, :tensor.size(1)] = tensor

        return pad_list

    def _print(self, *args):
        if not self.quiet:
            print(*args)
