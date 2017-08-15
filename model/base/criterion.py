import torch
import torch.nn as nn


def get_criterion_weights(n_words, PAD_token, ignore_pad=True):
    weights = torch.Tensor(n_words)
    weights[:] = 1
    if ignore_pad:
        weights[PAD_token] = 0

    return weights


def get_criterion(n_words, PAD_token, ignore_pad=True):
    weights = get_criterion_weights(n_words, PAD_token, ignore_pad=ignore_pad)
    criterion = nn.NLLLoss(weight=weights)
    criterion.ignore_pad = ignore_pad

    return criterion
