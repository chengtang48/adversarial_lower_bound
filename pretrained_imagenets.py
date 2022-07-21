# larger pretrained nets loader
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append("/Users/chengtang/Documents/Cheng-2021/tangch30-github/robust-learning/pytorch-playground/imagenet")

from vgg import vgg11, vgg16, cfg
from dataset import get

from medium_size_networks import *
from fast_appx import find_last_layer_matrix


def get_vgg11_hidden_layer_lower_bound(model, X, eps):
    L, U = create_init_bounds(X, eps)
    L, U = torch.tensor(np.array(L).reshape(X.shape[0], 3, X.shape[2], X.shape[3])), \
           torch.tensor(np.array(U).reshape(X.shape[0], 3, X.shape[2], X.shape[3]))

    conv_layer_cnt = 0
    model_params = list(model.parameters())
    assert len(model_params) == 22

    # feature
    X = torch.tensor(X)
    for v in cfg["A"]:
        if v == "M":
            # max pooling, kernel_size=2, stride=2
            L, U = fast_pool2D_layer_matrix_form(2, (2, 0), L, U, op='max')
        else:
            layer_idx = conv_layer_cnt*2
            W_k, b_k = model_params[layer_idx], model_params[layer_idx+1]
            #print(layer_idx, W_k.shape, b_k.shape)
            L, U = fast_conv2D_layer_matrix_form((W_k, b_k), (1, 1), L, U, activation=nn.ReLU())
            conv_layer_cnt += 1

    L, U = L.view(L.size(0), -1), U.view(U.size(0), -1)

    # classifier
    for i in range(-6, -2, 2):
        W, b = model_params[i], model_params[i+1]
        L, U = fast_layer_matrix_form((W, b), L, U, activation=nn.ReLU())

    # pre-softmax layer
    W_o, b_o = model_params[-2], model_params[-1]
    return L, U, W_o, b_o




def get_vgg11_hidden_layer_lower_bound(model, X, eps):
    L, U = create_init_bounds(X, eps)
    L, U = torch.tensor(np.array(L).reshape(X.shape[0], 3, X.shape[2], X.shape[3])), \
           torch.tensor(np.array(U).reshape(X.shape[0], 3, X.shape[2], X.shape[3]))

    conv_layer_cnt = 0
    model_params = list(model.parameters())
    assert len(model_params) == 22

    # feature
    X = torch.tensor(X)
    for v in cfg["A"]:
        if v == "M":
            # max pooling, kernel_size=2, stride=2
            L, U = fast_pool2D_layer_matrix_form(2, (2, 0), L, U, op='max')
        else:
            layer_idx = conv_layer_cnt*2
            W_k, b_k = model_params[layer_idx], model_params[layer_idx+1]
            #print(layer_idx, W_k.shape, b_k.shape)
            L, U = fast_conv2D_layer_matrix_form((W_k, b_k), (1, 1), L, U, activation=nn.ReLU())
            conv_layer_cnt += 1

    L, U = L.view(L.size(0), -1), U.view(U.size(0), -1)

    # classifier
    for i in range(-6, -2, 2):
        W, b = model_params[i], model_params[i+1]
        L, U = fast_layer_matrix_form((W, b), L, U, activation=nn.ReLU())

    # pre-softmax layer
    W_o, b_o = model_params[-2], model_params[-1]
    return L, U, W_o, b_o


def get_vgg16_hidden_layer_lower_bound(model, X, eps):
    # TODO: batchify, test
    L, U = create_init_bounds(X, eps)
    # L, U = torch.tensor(np.array(L).reshape(1, 3, X.shape[1], X.shape[2])), \
    #        torch.tensor(np.array(U).reshape(1, 3, X.shape[1], X.shape[2]))
    L, U = torch.tensor(np.array(L).reshape(X.shape[0], 3, X.shape[2], X.shape[3])), \
           torch.tensor(np.array(U).reshape(X.shape[0], 3, X.shape[2], X.shape[3]))

    conv_layer_cnt = 0
    model_params = list(model.parameters())
    assert len(model_params) == 32

    # feature
    X = torch.tensor(X)
    for v in cfg["D"]:
        if v == "M":
            # max pooling, kernel_size=2, stride=2
            L, U = fast_pool2D_layer_matrix_form(2, (2, 0), L, U, op='max')
        else:
            layer_idx = conv_layer_cnt*2
            W_k, b_k = model_params[layer_idx], model_params[layer_idx+1]
            #print(layer_idx, W_k.shape, b_k.shape)
            L, U = fast_conv2D_layer_matrix_form((W_k, b_k), (1, 1), L, U, activation=nn.ReLU())
            conv_layer_cnt += 1

    L, U = L.view(L.size(0), -1), U.view(U.size(0), -1)
    #L, U = L.reshape(-1), U.reshape(-1)
    # classifier
    for i in range(-6, -2, 2):
        W, b = model_params[i], model_params[i+1]
        L, U = fast_layer_matrix_form((W, b), L, U, activation=nn.ReLU())

    # pre-softmax layer
    W_o, b_o = model_params[-2], model_params[-1]

    return L, U, W_o, b_o


if __name__ == "__main__":
    # v_model = vgg16(pretrained=True,
    #                 model_root="/Users/chengtang/Documents/Cheng-2021/tangch30-github/robust-learning/adv-ReLU/models/vgg")

    v_model = vgg11(pretrained=True,
                    model_root="/Users/chengtang/Documents/Cheng-2021/tangch30-github/robust-learning/adv-ReLU/models/vgg")

    for param in v_model.parameters():
        print(param.data.shape)

    v_model.eval()

    ### test 1: batchified ibp
    from utils import regroup_data, batch_verify
    datapath = "/Users/chengtang/Documents/Cheng-2021/tangch30-github/robust-learning/adv-ReLU/data/temp"
    datadict = regroup_data(v_model, datapath, 30, 30, is_test=True)
    dataset = []
    for _, data in datadict.items():
        dataset.extend(data)

    print("val set has {} points".format(len(dataset)))

    #X, y = zip(*datadict[781])
    #X, y = np.array(X), np.array(y)
    #print(X.shape, y.shape)

    res = batch_verify(datadict, v_model, get_vgg11_hidden_layer_lower_bound, batch_size=10, eps=0.0001)
    safe_idx, acc_idx = zip(*res)
    safe_idx, acc_idx = np.array(safe_idx), np.array(acc_idx)
    safe_idx.astype(int)
    print("number of safe/total {}/{}".format(np.sum(safe_idx), len(safe_idx)))

    # for batch in val_set:
    #     # TODO: Add data transformation op
    #     img, target = batch
    #     # print("image shape", img.shape)
    #     preds = v_model(torch.tensor(img)).detach()
    #     # print(img.shape, target.shape)
    #     L, U, W_o, b_o = get_vgg11_hidden_layer_lower_bound(v_model, img, 0.00001) # 0.0000008(v16/50), 0.000001(v11/500)
    #     for i in range(len(target)):
    #         outp, y = preds[i], target[i]
    #         max_idx = np.argmax(outp)
    #         L_f = find_last_layer_bound(W_o, b_o, max_idx, L[i], U[i])
    #         if np.min(L_f.detach().numpy()) < 0:
    #             total_success += 1
    #             #print("success")
    #         # else:
    #         #     #print("failed")
    #         n_samples += 1
    #
    # print("success / total: {} / {}".format(total_success, n_samples))

    ### TODO: test 2 ibp-linear






