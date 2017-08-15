import time

from model import load_model
from model.trainer import Trainer

from util import arg_parser_eval as arg_parser
from util.util import config, bcolors
from util.time import as_minutes

# from data_handler.logistic_parser import LogisticDataParser as DataParser
from data_handler.handler import read_dataset


def get_loss_acc(trainer, data_parser, eval_data, args):
    acc_time_start = time.time()
    eval_acc = trainer.eval_acc(data_parser, eval_data, args.max_len)
    acc_time = time.time() - acc_time_start

    eval_loss = trainer.eval_loss(data_parser, eval_data, save_loss=False)
    return eval_acc, eval_loss, acc_time


def print_errors(trainer, data_parser, eval_data, args):
    test_acc, errors = trainer.eval_acc(data_parser, eval_data, max_length=args.max_len, get_errors=True)
    for error in errors:
        print('= %s' % ' '.join(error[0]))
        print('> %s' % ' '.join(error[1]))
        print('< %s' % ' '.join(error[2]))
        print()


def print_times(train_time, val_time, test_time):
    passed_time = as_minutes(test_time)
    print('Test Time: %s%s%s' % (bcolors.OKGREEN, passed_time, bcolors.ENDC))
    passed_time = as_minutes(train_time + val_time + test_time)
    print('Total Time: %s%s%s' % (bcolors.OKGREEN, passed_time, bcolors.ENDC))


def main(args):
    train_data, val_data, test_data, data_parser = read_dataset(args.data, cuda=args.cuda)
    args.max_len = min(args.max_len, data_parser.output_max_len)
    input_dict, output_dict = data_parser.input_dict, data_parser.output_dict

    net = load_model(args.load_model, input_dict, output_dict, load_last=args.load_last, eval=True, cuda=args.cuda)
    trainer = Trainer(net, cuda=args.cuda, batch_size=args.batch_size)

    if args.print_errors:
        print_errors(trainer, data_parser, test_data, args)

    train_acc, train_loss, train_time = get_loss_acc(trainer, data_parser, train_data, args)
    print('%sTrain acc %.6f%s' % (bcolors.OKBLUE, train_acc, bcolors.ENDC))
    print('%sTrain Loss %.6f%s' % (bcolors.OKBLUE, train_loss, bcolors.ENDC))

    val_acc, val_loss, val_time = get_loss_acc(trainer, data_parser, val_data, args)
    print('%sEval acc %.6f%s' % (bcolors.MAGENTA, val_acc, bcolors.ENDC))
    print('%sEval Loss %.6f%s' % (bcolors.MAGENTA, val_loss, bcolors.ENDC))

    test_acc, test_loss, test_time = get_loss_acc(trainer, data_parser, test_data, args)
    print('%sTest acc %.6f%s' % (bcolors.CYAN, test_acc, bcolors.ENDC))
    print('%sTest Loss %.6f%s' % (bcolors.CYAN, test_loss, bcolors.ENDC))

    if args.eval_time:
        print_times(train_time, val_time, test_time)

    # trainer.evaluate_randomly(data_parser, val_data, args.max_len, n=2)


if __name__ == '__main__':
    args = arg_parser.parse_args()
    assert args.load_model is not None, 'load_model is a necessary parameter'
    config(args)

    main(args)
