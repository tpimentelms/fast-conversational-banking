import random
import time
import sys
import math
import pickle
import torch
from torch import optim

from util.util import bcolors, signal_handler
from util.time import time_since


class Trainer(object):
    def __init__(self, net, optim_type='sgd', print_every=None, cuda=False, save_dir=None, save_every=1, improve_wait=500, batch_size=1):
        super(Trainer, self).__init__()

        self.net = net
        self.optim_type = optim_type
        self.improve_wait = improve_wait
        self.improve_wait_acc = False
        self.batch_size = batch_size
        
        self.print_every = print_every
        self.cuda = cuda
        self.save_dir = save_dir
        self.save_every = save_every

        self.iteration = 0
        self.print_loss_total = 0
        self.start = None
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.best_val_epoch = 0

        self.terminate = False

        signal_handler(self.handle_sigint_creator())

    def handle_sigint_creator(self):
        def handle_sigint(signal, frame):
            if not self.terminate:
                print('Received ctrl+c terminating execution at end of this epoch. Press CTRL+C again to terminate now')
                self.terminate = True
            else:
                print('Received second ctrl+c terminating execution now')
                sys.exit(0)
        return handle_sigint

    def train(self, input_variable, target_variable):
        self.optimizer.zero_grad()

        loss = self.net.full_forward(input_variable, target_variable)

        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def train_epoch(self, data_parser, train_data):
        self.net.train()
        epoch_size = len(train_data)
        epoch_loss = 0
        random.shuffle(train_data)

        for i in range(0, epoch_size, self.batch_size):
            input_variable, target_variable, tgt_len = self.get_batch(train_data, i, data_parser)
            loss = self.train(input_variable, target_variable)
            if math.isnan(loss):
                print('%sError: Received NaN value as loss, finishing training.%s' % (bcolors.FAIL, bcolors.ENDC))
                self.terminate = True
                break

            self.iteration += 1
            self.print_loss_total += loss / tgt_len
            epoch_loss += loss / tgt_len

            if self.print_every and self.iteration % self.print_every == 0:
                self.print_progress()

        return epoch_loss * self.batch_size / epoch_size

    def run_epoch(self, data_parser, train_data, val_data):
        loss = self.train_epoch(data_parser, train_data)
        eval_loss = self.eval_loss(data_parser, val_data)

        print('%s(%d/%d) Train Loss %.6f%s' % (bcolors.OKBLUE, self.epoch + 1, self.best_val_epoch + self.improve_wait + 1, loss, bcolors.ENDC))
        print('%s(%d/%d) Eval Loss %.6f\tBest eval loss %.6f%s' % (bcolors.OKGREEN, self.epoch + 1, self.best_val_epoch + self.improve_wait + 1, eval_loss, self.best_val_loss, bcolors.ENDC))

        self._save_checkpoint(data_parser, val_data, eval_loss)

    def _train_epochs(self, data_parser, train_data, val_data, start_epoch, n_epochs, lr=0.01, weight_decay=0.0):
        self.epoch_size = len(train_data)
        self.num_iterations = math.ceil(self.epoch_size / self.batch_size) * n_epochs

        self.optimizer = self.get_optimizer(lr=lr, weight_decay=weight_decay)

        self.epoch = start_epoch
        self.n_epochs = n_epochs
        for self.epoch in range(start_epoch, int(n_epochs)):
            self.run_epoch(data_parser, train_data, val_data)
            if self.terminate:
                return

        if self.epoch < n_epochs and n_epochs % 1 > 0:
            random.shuffle(train_data)
            train_data = train_data[:int(self.epoch_size * (n_epochs % 1))]
            self.run_epoch(data_parser, train_data, val_data)

    def train_epochs(self, data_parser, train_data, val_data, n_epochs, lr=0.01, weight_decay=0.0):
        self.reset_train_vars()
        self._train_epochs(data_parser, train_data, val_data, 0, n_epochs, lr=lr, weight_decay=weight_decay)

    def continue_training(self, data_parser, train_data, val_data, n_epochs, lr=0.01, weight_decay=0.0):
        self.epoch += 1
        self._train_epochs(data_parser, train_data, val_data, self.epoch, n_epochs, lr=lr, weight_decay=weight_decay)

    def eval_loss(self, data_parser, eval_data, save_loss=True):
        self.net.eval()
        loss_total = 0

        for i in range(0, len(eval_data), self.batch_size):
            input_variable, target_variable, tgt_len = self.get_batch(eval_data, i, data_parser)
            loss = self.net.full_forward(input_variable, target_variable)

            loss_total += loss.data[0] / tgt_len

        loss_avg = loss_total * self.batch_size / len(eval_data)

        if save_loss:
            if loss_avg < self.best_val_loss:
                self.best_val_loss = loss_avg
                self.best_val_epoch = self.epoch
            elif not self.improve_wait_acc and self.best_val_epoch + self.improve_wait <= self.epoch:
                print('%s(%d/%d) Eval Loss %.6f%s' % (bcolors.OKGREEN, self.epoch + 1, self.n_epochs, loss_avg, bcolors.ENDC))
                print('%sWarning: Haven\'t improved in %d epochs, finishing training.%s' % (bcolors.WARNING, self.improve_wait, bcolors.ENDC))
                self.terminate = True

        return loss_avg

    def check_acc(self, data_parser, src, tgt, max_length=None):
        max_length = max_length if max_length is not None else 100
        input_variable = data_parser.variable_from_sentence(data_parser.input_dict, src)
        input_variable = input_variable.transpose(0, 1)
        pred = self.net.translate(input_variable, data_parser, max_len=max_length, append_eos=False)
        return tgt == pred  # Check if equal without <eos>

    def eval_acc(self, data_parser, eval_data, max_length=None, get_errors=False, save_acc=False):
        self.net.eval()
        acc_total = 0
        errors = []

        for sample in eval_data:
            acc = self.check_acc(data_parser, sample[0], sample[1], max_length)
            if acc:
                acc_total += 1
            elif get_errors:
                pred = self.evaluate(data_parser, sample[0], max_length)
                errors += [(sample[0], sample[1], pred)]

        acc_avg = acc_total * 1.0 / len(eval_data)

        if save_acc:
            if acc_avg > self.best_val_acc:
                self.best_val_acc = acc_avg
                self.best_val_epoch = self.epoch
            elif self.improve_wait_acc and self.best_val_epoch + self.improve_wait <= self.epoch:
                print('%s(%d/%d) Eval Acc %.6f%s' % (bcolors.CYAN, self.epoch + 1, self.n_epochs, acc_avg, bcolors.ENDC))
                print('%sWarning: Haven\'t improved in %d epochs, finishing training.%s' % (bcolors.WARNING, self.improve_wait, bcolors.ENDC))
                self.terminate = True

        if get_errors:
            return acc_avg, errors
        return acc_avg

    def evaluate(self, data_parser, sentence, max_length):
        input_variable = data_parser.variable_from_sentence(data_parser.input_dict, sentence)
        input_variable = input_variable.transpose(0, 1)
        decoded_words = self.net.translate(input_variable, data_parser, max_len=max_length)

        return decoded_words

    def evaluate_randomly(self, data_parser, lang_pairs, max_length, n=10):
        print('%sRandom translations%s' % (bcolors.MAGENTA, bcolors.ENDC))
        for i in range(n):
            pair = random.choice(lang_pairs)
            print('>', ' '.join(pair[0]))
            print('=', ' '.join(pair[1]))
            output_words = self.evaluate(data_parser, pair[0], max_length)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    def get_optimizer(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        if self.optim_type == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.optim_type == 'momentum':
            optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif self.optim_type == 'adam':
            optimizer = optim.Adam(self.net.parameters(), weight_decay=weight_decay)
        elif self.optim_type == 'rmsprop':
            optimizer = optim.RMSprop(self.net.parameters(), weight_decay=weight_decay)
        else:
            raise NotImplementedError('Optimizers available are sgd|momentum|adam|rmsprop. %s is not an option.' % (self.optim_type))
        return optimizer

    def get_batch(self, data, i, data_parser):
        return self.separate_data_pairs(data[i:i + self.batch_size], data_parser)
        
    def separate_data_pair(self, data_pair, data_parser):
        src_var, tgt_var = data_parser.variables_from_pair(data_pair)
        return src_var, tgt_var

    def separate_data_pairs(self, data_pairs, data_parser):
        src_var, tgt_var, tgt_avg_len = data_parser.variables_from_pairs(data_pairs)
        return src_var, tgt_var, tgt_avg_len

    def reset_train_vars(self):
        self.start = time.time()
        self.iteration = 0
        self.print_loss_total = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0

    def print_progress(self):
        temp_epochs = min(self.best_val_epoch + self.improve_wait + 1, self.n_epochs)
        self.num_iterations = math.ceil(self.epoch_size / self.batch_size) * temp_epochs

        print_loss_avg = self.print_loss_total / self.print_every
        self.print_loss_total = 0

        print('%2.2f%%: Iteration %d/%d | Loss %.4f   (%s)' % (
            self.iteration / self.num_iterations * 100, self.iteration, self.num_iterations, print_loss_avg,
            time_since(self.start, self.iteration / self.num_iterations)))

    def _save_checkpoint(self, data_parser, val_data, loss):
        if self.epoch % self.save_every != 0:
            return

        if self.improve_wait_acc:
            val_acc = self.eval_acc(data_parser, val_data, save_acc=True)
            print('%s(%d/%d) Eval ACC %.4f\tBest eval ACC %.4f%s' % (bcolors.CYAN, self.epoch + 1, self.best_val_epoch + self.improve_wait + 1, val_acc, self.best_val_acc, bcolors.ENDC))
            self.save_checkpoint(val_acc, best_loss=False)
        else:
            self.save_checkpoint(loss)

    def save_checkpoint(self, loss, best_loss=True):
        if not self.save_dir:
            return

        self.save_train_data(self.save_dir + '/trainer.pckl')
        self.net.save_config_data(self.save_dir + '/model.pckl')
        self.save_model(self.net, self.save_dir + '/last.pyth7')
        if (best_loss and loss <= self.best_val_loss) or \
                (not best_loss and loss >= self.best_val_acc):
            self.save_model(self.net, self.save_dir + '/best.pyth7')

    def save_train_data(self, path):
        checkpoint_data = {
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_val_epoch': self.best_val_epoch,
            'iteration': self.iteration,
            'start': self.start,
            'time_passed': time.time() - self.start,
            'optim': self.optim_type,
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint_data, f, -1)

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    # Model should be previously initialized. Else: use model.load_model().
    def load_checkpoint(self, load_dir):
        self.load_train_data(load_dir + '/trainer.pckl')
        self.load_model(load_dir + '/last.pyth7')

    def load_train_data(self, path):
        with open(path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        self.epoch = checkpoint_data['epoch']
        self.best_val_loss = checkpoint_data['best_val_loss']
        self.best_val_acc = checkpoint_data['best_val_acc']
        self.best_val_epoch = checkpoint_data['epoch']
        self.iteration = checkpoint_data['iteration']

        self.start = time.time() - checkpoint_data['time_passed']

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))
        # self.net.load_state_dict(torch.load(path), map_location=lambda storage, loc: storage)
