import random
import os
import torch
import signal


def signal_handler(function):
    signal.signal(signal.SIGINT, function)


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def config(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if 'save_dir' in args and args.save_dir:
        mkdir(args.save_dir)


class bcolors:
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
