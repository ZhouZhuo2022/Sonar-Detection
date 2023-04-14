import torch


def wh(x):
    if torch.is_tensor(x):
        h, w = x.shape[-2:]
    else:
        w, h = x.size
    return w, h


def hw(x):
    if torch.is_tensor(x):
        h, w = x.shape[-2:]
    else:
        w, h = x.size
    return h, w
