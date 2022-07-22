# linear outer approximation from intermediate layer
# note: currently doesn't support batch op

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_appx import fast_layer_matrix_form, fast_ibp_layer, act_fn_mapping


def construct_appx_params(L, U, activation_type="relu"):
    """
    L, U are pre-activation bounds from ibp
    :param L: shape = (d_out,)
    :param U:
    :return:
    """
    d_L, d_U = torch.tensor(np.zeros(len(L))), torch.tensor(np.zeros(len(L)))
    b_L, b_U = torch.tensor(np.zeros(len(L))), torch.tensor(np.zeros(len(L)))
    if activation_type == "relu":
        d_L[L > 0] = 1
        d_U[L > 0] = 1
        ind = (L < 0) & (0 < U)
        if np.any(ind.numpy()):
            d_L[ind] = torch.div(U[ind], U[ind] - L[ind])
            d_U = d_L
            b_U[ind] = torch.div(U[ind], U[ind] - L[ind]) * -L[ind]
    #print(L, U)
    #print(d_U, b_U)
    return d_L, d_U, b_L, b_U


def update_opt_params(prev_params, network_params, ibp_bounds, is_first_layer=False):
    """
    :param prev_params: (modified) network params at layer l
    :param network_params: network params at layer l-1
    :param ibp_bounds: cached pre-activated ibp bounds at layer l-1
    :return: modified network params at layer l-1
    """
    L, U = ibp_bounds # prev-act ibp bounds from layer l-1
    W, b, lin_op_type, act_type = network_params # params of layer l-1
    d_L, d_U, b_L, b_U = construct_appx_params(L, U, activation_type=act_type)
    W_prev_L, W_prev_U, b_prev_L, b_prev_U, lin_op_type_prev = prev_params # params of layer l
    P_L, P_U, W_L, W_U = None, None, None, None

    if lin_op_type_prev == 'linear' or not is_first_layer:
        # linear operation type at layer l
        P_L = W_prev_L * d_U + F.relu(W_prev_L) * (d_L - d_U)
        P_U = W_prev_U * d_U + F.relu(-1*W_prev_U) * (d_U - d_L)

    if lin_op_type == "linear":
        # linear op type at layer l-1
        # TODO: need to check if these implementation are correct
        W_L = torch.matmul(P_L, W)
        b_L_ = torch.matmul(P_L, b) - torch.matmul(F.relu(-1*W_prev_L), b_U) + torch.matmul(F.relu(W_prev_L), b_L) + b_prev_L
        W_U = torch.matmul(P_U, W)
        b_U_ = torch.matmul(P_U, b) + torch.matmul(F.relu(W_prev_U), b_U) - torch.matmul(F.relu(-1*W_prev_U), b_L) + b_prev_U

    return (W_L, W_U, b_L_, b_U_, lin_op_type)


######
def run_ibp_with_linear(hidden_network_params, linear_appx_layer_schedule, L_0, U_0):
    """
    runs ibp forward with occasional linear approximation based on linear_appx_layer_schedule
    :param hidden_network_params:
    :param linear_appx_layer_schedule:
    :param L_0: torch tensor
    :param U_0:
    :return:
    """
    L, U = L_0, U_0
    cached_pre_ibp_bounds = None
    if len(linear_appx_layer_schedule) > 0:
        cached_pre_ibp_bounds = []
        lin_appx_start_ids, backward_steps = list(zip(*linear_appx_layer_schedule))

    lin_appx_iter = 0
    for layer_idx, network_param in enumerate(hidden_network_params):
        has_lin_appx = len(linear_appx_layer_schedule) > 0 and lin_appx_iter < len(lin_appx_start_ids)
        if has_lin_appx:
            L, U, L_pre, U_pre = fast_ibp_layer(network_param, L, U, return_pre_act=True)
            if layer_idx <= max(lin_appx_start_ids):
                cached_pre_ibp_bounds.append((L_pre, U_pre))
            if has_lin_appx and layer_idx == lin_appx_start_ids[lin_appx_iter]:
                # run backward linear approximation
                k = backward_steps[lin_appx_iter]
                print("running lin-appx at layer {} for {} backward steps".format(layer_idx, k))
                net_param, lin_op_type, act_type, conv_param = network_param # l
                if lin_op_type == "linear":
                    W, b = net_param
                    prev_params = (W, W, b, b, "linear")
                else:
                    # TODO: add conv and pool layers
                    print("unsupported op")
                # l to l - k + 1
                for idx in range(layer_idx - 1, layer_idx - k):
                    net_param, lin_op_type, act_type, conv_param = hidden_network_params[idx]
                    if lin_op_type == "linear":
                        W, b = net_param
                        curr_params = W, b, lin_op_type, act_type
                    else:
                        # TODO: add conv and pool layers
                        print("unsupported op")
                    prev_params = update_opt_params(prev_params, curr_params, cached_pre_ibp_bounds[idx])

                # update pre-act ibp bounds for chosen layer (at layer_idx)
                L_0, U_0 = cached_pre_ibp_bounds[layer_idx - k] # input ibp pre-act bounds at layer l-k+1
                _, _, act_type, _ = hidden_network_params[layer_idx - k]
                act_fn = act_fn_mapping[act_type]
                L_0, U_0 = act_fn(L_0), act_fn(U_0) # input ibp bounds at layer l-k+1
                W_L, W_U, b_L, b_U, lin_op_type = prev_params # modified param for layer l
                #
                L_tilde, _ = fast_ibp_layer(((W_L, b_L), lin_op_type, None, None), L_0, U_0)
                _, U_tilde = fast_ibp_layer(((W_U, b_U), lin_op_type, None, None), L_0, U_0)
                L_hat, U_hat = cached_pre_ibp_bounds[layer_idx]
                #print(L_tilde, L_hat, torch.max(L_tilde, L_hat))
                #print(U_tilde, U_hat, torch.min(U_tilde, U_hat))
                L_pre, U_pre = torch.max(L_tilde, L_hat), torch.min(U_tilde, U_hat) # local linear pre-act appx for layer l
                cached_pre_ibp_bounds[layer_idx] = L_pre, U_pre
                ## update activated bounds at layer
                _, _, act_type, _ = hidden_network_params[layer_idx]
                act_fn = act_fn_mapping[act_type]
                L, U = act_fn(L_pre), act_fn(U_pre)
                lin_appx_iter += 1
        else:
            L, U = fast_ibp_layer(network_param, L, U)
    return L, U


if __name__ == "__main__":
    import numpy as np
    # ### test 1: single layer ibp is better than single layer linear appx for simple relu-linear layer
    # np.random.seed(0)
    # d_out, d_in = 200, 200
    # W, b = torch.tensor(np.random.randn(d_out, d_in)), torch.tensor(np.random.randn(d_out))
    # network_param = (W, b, "linear", "relu") # layer l
    # # init random hyper-rectangular bounds after layer l-1
    # L_0 = np.random.randn(d_in)
    # eps = 0.1
    # U_0 = L_0 + eps
    # L_0, U_0 = torch.tensor(L_0), torch.tensor(U_0)
    # L_hat, U_hat = fast_layer_matrix_form((W, b), L_0, U_0, activation=None) # pre-act ibp bounds
    # L_ibp, U_ibp = F.relu(L_hat), F.relu(U_hat) # ibp for layer l
    # # identity layer at l+1 (no effect)
    # prev_params = (torch.tensor(np.eye(d_out, d_out)), torch.tensor(np.eye(d_out, d_out)), torch.tensor(np.zeros(d_out)), torch.tensor(np.zeros(d_out)), "linear")
    # params = update_opt_params(prev_params, network_param, (L_hat, U_hat))
    # W_L, W_U, b_L, b_U, _ = params
    # assert np.all(W_L.numpy() - W_U.numpy() == 0)
    # L_lin, _ = fast_layer_matrix_form((W_L, b_L), L_0, U_0, activation=None)
    # _, U_lin = fast_layer_matrix_form((W_U, b_U), L_0, U_0, activation=None)
    # #print((L_ibp - L_lin).numpy())
    # #print((U_ibp-U_lin).numpy())
    # assert np.all((L_ibp - L_lin).numpy() >= 0) and np.sum((U_ibp-U_lin).numpy() == 0)

    ### test 2: two-layer ibp
    np.random.seed(0)
    d_out, d_in, d_fin = 10, 10, 5
    W, b = torch.tensor(np.random.randn(d_out, d_in)), torch.tensor(np.random.randn(d_out))
    network_param = (W, b, "linear", "relu")  # layer l
    # init random hyper-rectangular bounds after layer l-1
    L_0 = np.random.randn(d_in)
    eps = 0.1
    U_0 = L_0 + eps
    L_0, U_0 = torch.tensor(L_0), torch.tensor(U_0)
    L_hat, U_hat = fast_layer_matrix_form((W, b), L_0, U_0, activation=None)  # pre-act ibp bounds
    L_ibp, U_ibp = F.relu(L_hat), F.relu(U_hat)  # ibp for layer l
    # identity layer at l+1 (no effect)
    np.random.seed(1)
    W_pre, b_pre = torch.tensor(np.random.rand(d_fin, d_out)), torch.tensor(np.zeros(d_fin))
    prev_params = (W_pre, W_pre, b_pre, b_pre, "linear")
    params = update_opt_params(prev_params, network_param, (L_hat, U_hat))
    W_L, W_U, b_L, b_U, _ = params
    assert np.all(W_L.numpy() - W_U.numpy() == 0)
    L_lin, _ = fast_layer_matrix_form((W_L, b_L), L_0, U_0, activation=None)
    _, U_lin = fast_layer_matrix_form((W_U, b_U), L_0, U_0, activation=None)
    L_lin, U_lin = F.relu(L_lin), F.relu(U_lin)
    L_ibp, U_ibp = fast_layer_matrix_form((W_pre, b_pre), L_ibp, U_ibp, activation=nn.ReLU())
    #L_ibp, U_ibp = fast_layer_matrix_form((W_pre, b_pre), L_ibp, U_ibp, activation=None)
    print((L_ibp - L_lin).numpy())
    print((U_ibp-U_lin).numpy())


    ### test 3: single layer ibp is better than single layer linear appx for relu-convolution layer




