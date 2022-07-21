import numpy as np
import torch
from tqdm import tqdm

from dataset import get
from fast_appx import *


def regroup_data(model, datapath, batch_size, max_n_batch, is_test=False):
    model.eval()
    data_dict = dict()
    n_batch = 0
    for batch in tqdm(get(batch_size, data_root=datapath, train=not is_test, val=is_test)):
        X, y = batch
        preds = model(torch.tensor(X))
        yhat = np.argmax(preds.detach().numpy(), axis=1)
        n_batch += 1
        for xi, yhati, yi in zip(X, yhat, y):
            if yhati in data_dict:
                data_dict[yhati].append((xi, yi))
            else:
                data_dict[yhati] = [(xi, yi)]
        if n_batch > max_n_batch:
            break
    return data_dict


def batch_verify(datadict, model, bound_fn, batch_size=10, eps=0.0001):
    # TODO: test
    model.eval()
    res = []
    for yhat, data in datadict.items():
        X, y = zip(*data)
        X = np.array(X)
        y = np.array(y)
        idx = 0
        # batchify examples with prediction yhat
        while idx < len(y):
            print("processing batch {}, yhat idx {}".format(idx//batch_size, yhat))
            batch = X[idx:idx+batch_size]
            # print("batch sizes", batch.shape)
            L, U, W_o, b_o = bound_fn(model, batch, eps)
            W_L, b_L = find_last_layer_matrix(W_o, b_o, yhat) # same obj matrix for same yhat
            L, _ = fast_layer_matrix_form((W_L, b_L), L, U, activation=None)
            L_f = np.min(L.detach().numpy(), axis=1)
            safe_idx = L_f >= 0
            acc_idx = y == yhat
            res.extend(list(zip(safe_idx, acc_idx)))
            idx += batch_size
    return res


def binsearch(datadict, model, bound_fn, batch_size=10, eps0=0.1, timeout=1000):
    # TODO: perhaps search by group ??
    eps_curr = eps0
    n_safe_pre, n_safe_curr = 0, 0
    inc_rate = 0.1
    while n_safe_pre == 0 or (n_safe_curr - n_safe_pre) / n_safe_pre > inc_rate:
        n_safe_pre = n_safe_curr
        eps_curr /= 2
        # add timer
        res = batch_verify(datadict, model, bound_fn, batch_size=batch_size, eps=eps_curr)
        safe_idx, acc_idx = zip(*res)
        n_safe_curr = np.sum(safe_idx)
    return eps_curr

