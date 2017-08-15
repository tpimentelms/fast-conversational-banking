from util import arg_parser_train as arg_parser
from util.util import config, bcolors
from data_handler.handler import read_dataset

from model import get_model
from model.trainer import Trainer


def train(train_data, val_data, data_parser, args):
    input_dict, output_dict = data_parser.input_dict, data_parser.output_dict

    net = get_model(
        input_dict, output_dict, args.model, args.hidden_size, args.max_len, args.layers, args.dropout,
        kernel_size=args.kernel_size, dilate=args.dilate, ignore_pad=args.ignore_pad,
        multilinear=(not args.single_linear), input_len=data_parser.input_max_len, stride=args.stride,
        attn_heads=args.attn_heads, cuda=args.cuda)
    net.initialize_params(args.init_params)

    print(args)
    print()
    print(net)
    print('Number of Elements: %d' % (sum([x.view(-1).size(0) for x in net.parameters()])))

    trainer = Trainer(net, optim_type=args.optim, print_every=args.print_every, cuda=args.cuda,
                      save_dir=args.save_dir, save_every=args.save_every, improve_wait=args.improve_wait,
                      batch_size=args.batch_size)
    if args.train_from:
        trainer.load_checkpoint(args.train_from)
        trainer.continue_training(data_parser, train_data, val_data, args.epochs, lr=args.lr, weight_decay=args.weight_decay)
    else:
        trainer.train_epochs(data_parser, train_data, val_data, args.epochs, lr=args.lr, weight_decay=args.weight_decay)

    return trainer


def main(args):
    train_data, val_data, test_data, data_parser = read_dataset(args.data, reduce_ratio=args.reduce_ratio, cuda=args.cuda)
    print('Train size: %d, Val size: %d, Test size: %d' % (len(train_data), len(val_data), len(test_data)))
    args.input_max_len = data_parser.input_max_len
    args.max_len = min(args.max_len, data_parser.output_max_len)

    trainer = train(train_data, val_data, data_parser, args)
    trainer.evaluate_randomly(data_parser, train_data, args.max_len, n=2)

    # trainer.eval_loss(data_parser, val_data)
    train_acc = trainer.eval_acc(data_parser, train_data, args.max_len)
    print('%sTrain acc %.6f%s' % (bcolors.CYAN, train_acc, bcolors.ENDC))
    val_acc = trainer.eval_acc(data_parser, val_data, args.max_len)
    print('%sEval acc %.6f%s' % (bcolors.CYAN, val_acc, bcolors.ENDC))
    val_acc = trainer.eval_acc(data_parser, test_data, args.max_len)
    print('%sTest acc %.6f%s' % (bcolors.CYAN, val_acc, bcolors.ENDC))

    trainer.evaluate_randomly(data_parser, val_data, args.max_len, n=2)


if __name__ == '__main__':
    args = arg_parser.parse_args()
    config(args)
    main(args)
