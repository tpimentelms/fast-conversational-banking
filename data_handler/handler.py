import pickle
import random
import math


def read_dataset(file_name, reduce_ratio=1.0, cuda=True):
    with open(file_name, 'rb') as f:
        train_data, val_data, test_data, data_parser = pickle.load(f)

    if reduce_ratio != 1.0:
        train_data = reduce_dataset(train_data, reduce_ratio)
    data_parser._cuda = cuda

    return train_data, val_data, test_data, data_parser


def write_dataset(train_data, val_data, test_data, data_parser, file_name):
    data = [train_data, val_data, test_data, data_parser]
    with open(file_name, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def get_eval_data(data_parser, src, tgt, train_data, val_size):
    if src is not None:
        eval_data = data_parser.read_data(src, tgt)
    else:
        eval_data = train_data[-val_size:]
        train_data = train_data[:-val_size]

    return train_data, eval_data


def reduce_dataset(train_data, reduce_ratio):
    if reduce_ratio < 1.0 and reduce_ratio > 0.0:
        random.shuffle(train_data)
        train_data = train_data[:int(math.floor(reduce_ratio * len(train_data)))]
    elif reduce_ratio <= 0.0 or reduce_ratio > 1.0:
        raise ValueError('reduce_ratio should be between zero and one \in(0,1]. Received %f' % reduce_ratio)
    return train_data
