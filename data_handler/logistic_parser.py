from .data_parser import DataParser


class LogisticDataParser(DataParser):
    def __init__(self, max_len, cuda=True, quiet=True, remove_brackets=False):
        super(LogisticDataParser, self).__init__(max_len, cuda=cuda, quiet=quiet)

        self.remove_brackets = remove_brackets

    def normalize_string(self, s):
        if self.remove_brackets:
            s = [x for x in s if x not in ['(', ')', ',']]

        return s
