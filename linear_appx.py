# ibp with linear outer approximation from any intermediate layer
# note: currently doesn't support batch processing
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_appx import fast_layer_matrix_form, fast_ibp_layer, act_fn_mapping
from conv2d_mat import conv2d_bias


def construct_appx_params(L, U, lin_op_type="linear", activation_type="relu"):
    """
    construct appx param matrix, bias
    :param L: pre-activation ibp bound of current layer (after linear op)
    :param U:
    :param lin_op_type: current layer linear op type
    :param activation_type: current layer activation type
    :return:
    """
    if lin_op_type == "conv2d":
        n_in, i_h, i_w = L.shape
        L, U = L.reshape(-1), U.reshape(-1)

    #d_L, d_U = torch.tensor(np.zeros(len(L))), torch.tensor(np.zeros(len(L)))
    #b_L, b_U = torch.tensor(np.zeros(len(L))), torch.tensor(np.zeros(len(L)))
    d_L, d_U = torch.zeros(len(L)), torch.zeros(len(L))
    b_L, b_U = torch.zeros(len(L)), torch.zeros(len(L))

    if activation_type == "relu":
        d_L[L > 0] = 1
        d_U[L > 0] = 1
        ind = (L < 0) & (0 < U)
        #if np.any(ind.numpy()):
        if torch.any(ind):
            d_L[ind] = torch.div(U[ind], U[ind] - L[ind])
            d_U = d_L
            #d_U[ind] = torch.div(U[ind], U[ind] - L[ind])
            # d_U = d_L
            b_U[ind] = torch.div(U[ind], U[ind] - L[ind]) * -L[ind]

    if lin_op_type == "conv2d":
        d_L, d_U = d_L.view(n_in, i_h, i_w), d_U.view(n_in, i_h, i_w)
        b_L, b_U = b_L.view(n_in, i_h, i_w), b_U.view(n_in, i_h, i_w)

    #print(L, U)
    #print(d_U, b_U)
    return d_L, d_U, b_L, b_U


def update_opt_params(prev_params, network_params, ibp_bounds, is_first_layer=False):
    """
    :param prev_params: (modified) network params at layer l
    :param network_params: network params at layer l-1
    :param ibp_bounds: cached pre-activated ibp bounds at layer l-1
    :return: modified network params at layer l-1
    TODO: add pool ops
    """
    L, U = ibp_bounds # prev-act ibp bounds from layer l-1
    net_param, lin_op_type, act_type, conv_param = network_params # params of layer l-1
    W, b = net_param
    W_prev_L, W_prev_U, b_prev_L, b_prev_U, lin_op_type_prev, conv_param_prev = prev_params # params of layer l
    P_L, P_U, W_L, W_U = None, None, None, None
    n_in, i_h, i_w = None, None, None
    # linear operation type at layer l
    if lin_op_type_prev == 'linear' or not is_first_layer:
        # regardless of op type, if l is not the first layer, then we apply simple linear op W_l shape = (d_out, d_{l-1})
        d_L, d_U, b_L, b_U = construct_appx_params(L, U, lin_op_type=lin_op_type, activation_type=act_type)
        if lin_op_type != "linear":
            n_in, i_h, i_w = d_U.size()
            d_L, d_U = d_L.reshape(-1), d_U.reshape(-1)
        P_L = W_prev_L * d_U + F.relu(W_prev_L) * (d_L - d_U)
        P_U = W_prev_U * d_U + F.relu(-1*W_prev_U) * (d_U - d_L)

    elif lin_op_type_prev == 'conv2d' and is_first_layer:
        # if op type is conv2d and l is the first layer, we use conv2d op to realize W_l shape=(d_out, d_{l-1})
        stride, padding, *params = conv_param_prev
        assert lin_op_type == "conv2d"
        d_L, d_U, b_L, b_U = construct_appx_params(L, U, lin_op_type=lin_op_type, activation_type=act_type)
        n_in, i_h, i_w = d_U.size()
        d_in = n_in*i_h*i_w
        inp1 = torch.eye(d_in, d_in) * d_U.view(d_in)
        inp2 = torch.eye(d_in, d_in) * (d_L - d_U).view(d_in)

        P_L = F.conv2d(inp1.view(d_in, n_in, i_h, i_w),
                       W_prev_L, stride=stride, padding=padding, *params)
        P_L += F.conv2d(inp2.view(d_in, n_in, i_h, i_w),
                        F.relu(W_prev_L), stride=stride, padding=padding, *params)
        P_U = F.conv2d(inp1.view(d_in, n_in, i_h, i_w),
                       W_prev_U, stride=stride, padding=padding, *params)
        P_U += F.conv2d(-1*inp2.view(d_in, n_in, i_h, i_w),
                        F.relu(-1*W_prev_U), stride=stride, padding=padding, *params)

    # linear op type at layer l-1
    # TODO: debug by cases
    # 1. prev is conv2d, curr is linear (not possible)
    # 2. prev is linear, curr is conv2d (need to reshape b_U, b_L?)
    # 3. prev is linear, curr is linear
    # 4. prev is conv2d, curr is conv2d
    if lin_op_type == "linear":
        W_L = torch.matmul(P_L, W)
        b_L_ = torch.matmul(P_L, b) - torch.matmul(F.relu(-1*W_prev_L), b_U) + torch.matmul(F.relu(W_prev_L), b_L) + b_prev_L
        W_U = torch.matmul(P_U, W)
        b_U_ = torch.matmul(P_U, b) + torch.matmul(F.relu(W_prev_U), b_U) - torch.matmul(F.relu(-1*W_prev_U), b_L) + b_prev_U

    elif lin_op_type == "conv2d":
        # if op at l-1 is conv2d, we apply conv2d_transpose to realize left multiplication with matrix W_{l-1}
        # W_{l-1} shape=(d_{l-1}, d_{l-2})
        stride, padding, *params = conv_param
        P_L = P_L.view(-1, n_in, i_h, i_w)
        P_U = P_U.view(-1, n_in, i_h, i_w)

        assert len(b) == n_in
        b = conv2d_bias(b, i_h, i_w)

        W_L = F.conv_transpose2d(P_L, W, stride=stride, padding=padding, *params)
        W_U = F.conv_transpose2d(P_U, W, stride=stride, padding=padding, *params)
        b_L_ = torch.matmul(P_L.view(-1, n_in*i_h*i_w), b)
        b_U_ = torch.matmul(P_U.view(-1, n_in * i_h * i_w), b)

        if lin_op_type_prev == "conv2d":
            b_L_ -= F.conv2d(b_U, F.relu(-1*W_prev_L), stride=stride, padding=padding, *params).view(-1)
            b_L_ += F.conv2d(b_L, F.relu(W_prev_L), stride=stride, padding=padding, *params).view(-1) + b_prev_L
            b_U_ += F.conv2d(b_U, F.relu(W_prev_U), stride=stride, padding=padding, *params).view(-1) + b_prev_U
            b_U_ -= F.conv2d(b_L, F.relu(-1*W_prev_U), stride=stride, padding=padding, *params).view(-1)
        elif lin_op_type_prev ==  "linear":
            b_L, b_U = b_L.view(-1), b_U.view(-1)
            b_L_ -= torch.matmul(F.relu(-1*W_prev_L), b_U)
            b_L_ += torch.matmul(F.relu(W_prev_L), b_L) + b_prev_L
            b_U_ += torch.matmul(F.relu(W_prev_U), b_U) + b_prev_U
            b_U_ -= torch.matmul(F.relu(-1*W_prev_U), b_L)

    return (W_L, W_U, b_L_, b_U_, lin_op_type, conv_param)


######
def run_ibp_with_linear(hidden_network_params, linear_appx_layer_schedule, L_0, U_0):
    """
    TODO: debug implementation
    runs ibp forward with occasional linear approximation based on linear_appx_layer_schedule
    :param hidden_network_params:
    :param linear_appx_layer_schedule:
    :param L_0: torch tensor
    :param U_0:
    :return:
    """
    L, U = L_0, U_0
    L_init, U_init = L_0, U_0
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
                assert layer_idx - k >= -1
                print("running lin-appx at layer {} for {} backward steps".format(layer_idx, k))
                net_param, lin_op_type, _, conv_param = network_param # l
                # if lin_op_type == "linear":
                #     W, b = net_param
                #     prev_params = (W, W, b, b, "linear")
                # else:
                #     # TODO: add conv and pool layers
                #     print("unsupported op")
                W, b = net_param
                prev_params = (W, W, b, b, lin_op_type, conv_param) # param at starting layer l
                # run backward linear appx from l to l - k + 1
                for idx in range(layer_idx - 1, layer_idx - k, -1):
                    # net_param, lin_op_type, act_type, conv_param = hidden_network_params[idx]
                    # if lin_op_type == "linear":
                    #     W, b = net_param
                    #     curr_params = W, b, lin_op_type, act_type
                    # else:
                    #     # TODO: add conv and pool layers
                    #     print("unsupported op")
                    prev_params = update_opt_params(prev_params,
                                                    hidden_network_params[idx],
                                                    cached_pre_ibp_bounds[idx],
                                                    is_first_layer= idx == layer_idx-1)

                ### update pre-act ibp bounds for chosen layer (at layer_idx)
                # get input ibp pre-act bounds at layer l-k+1
                if layer_idx - k == -1:
                    L_0, U_0 = L_init, U_init
                else:
                    L_0, U_0 = cached_pre_ibp_bounds[layer_idx - k]
                    _, _, act_type, _ = hidden_network_params[layer_idx - k]
                    act_fn = act_fn_mapping[act_type]
                    L_0, U_0 = act_fn(L_0), act_fn(U_0) # input ibp bounds at layer l-k+1
                # regardless of linear op type at layer l-k+1, the last step is always a simple linear op with modified
                # matrix W_{l-k+1}
                W_L, W_U, b_L, b_U, _, _ = prev_params # modified param for layer l
                #
                # L_tilde, _ = fast_ibp_layer(((W_L, b_L), lin_op_type, None, None), L_0, U_0)
                # _, U_tilde = fast_ibp_layer(((W_U, b_U), lin_op_type, None, None), L_0, U_0)
                L_tilde, _ = fast_layer_matrix_form((W_L, b_L), L_0, U_0)
                _, U_tilde = fast_layer_matrix_form((W_U, b_U), L_0, U_0)
                L_hat, U_hat = cached_pre_ibp_bounds[layer_idx]
                #print(L_tilde, L_hat, torch.max(L_tilde, L_hat))
                #print(U_tilde, U_hat, torch.min(U_tilde, U_hat))
                L_pre, U_pre = torch.max(L_tilde, L_hat), torch.min(U_tilde, U_hat) # local linear pre-act appx for layer l
                print(np.all((L_tilde - L_hat).numpy() > 0))
                #L_pre, U_pre = L_tilde, U_tilde
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
    #### test 1: single layer ibp is better than single layer linear appx for simple relu-linear layer
    # np.random.seed(0)
    # d_out, d_in = 200, 200
    # W, b = torch.tensor(np.random.randn(d_out, d_in)).float(), torch.tensor(np.random.randn(d_out)).float()
    # # network_param = (W, b, "linear", "relu") # layer l
    # network_param = ((W, b), "linear", "relu", None)
    # # init random hyper-rectangular bounds after layer l-1
    # L_0 = np.random.randn(d_in)
    # eps = 0.1
    # U_0 = L_0 + eps
    # L_0, U_0 = torch.tensor(L_0).float(), torch.tensor(U_0).float()
    # L_hat, U_hat = fast_layer_matrix_form((W, b), L_0, U_0, activation=None) # pre-act ibp bounds
    # L_ibp, U_ibp = F.relu(L_hat), F.relu(U_hat) # ibp for layer l
    # # identity layer at l+1 (no effect)
    # prev_params = (torch.eye(d_out, d_out), torch.eye(d_out, d_out),
    #                torch.zeros(d_out), torch.zeros(d_out), "linear", None)
    # params = update_opt_params(prev_params, network_param, (L_hat, U_hat))
    # W_L, W_U, b_L, b_U, _, _ = params
    # assert np.all(W_L.numpy() - W_U.numpy() == 0)
    # L_lin, _ = fast_layer_matrix_form((W_L, b_L), L_0, U_0, activation=None)
    # _, U_lin = fast_layer_matrix_form((W_U, b_U), L_0, U_0, activation=None)
    # #print((L_ibp - L_lin).numpy())
    # #print((U_ibp-U_lin).numpy())
    # assert np.all((L_ibp - L_lin).numpy() >= 0) and np.sum((U_ibp-U_lin).numpy() == 0)

    ### test 2: two-layer ibp
    np.random.seed(0)
    d_out, d_in, d_fin = 10, 10, 10
    W, b = torch.tensor(np.random.randn(d_out, d_in)).float(), torch.tensor(np.random.randn(d_out)).float()
    #network_param = (W, b, "linear", "relu")  # layer l
    network_param = ((W, b), "linear", "relu", None)
    # init random hyper-rectangular bounds after layer l-1
    L_0 = np.random.randn(d_in)
    eps = 0.1
    U_0 = L_0 + eps
    L_0, U_0 = torch.tensor(L_0).float(), torch.tensor(U_0).float()
    L_hat, U_hat = fast_layer_matrix_form((W, b), L_0, U_0, activation=None)  # pre-act ibp bounds
    L_ibp, U_ibp = F.relu(L_hat), F.relu(U_hat)  # ibp for layer l
    # identity layer at l+1 (no effect)
    np.random.seed(1)
    W_pre, b_pre = torch.tensor(np.random.rand(d_fin, d_out)).float(), torch.tensor(np.zeros(d_fin)).float()
    prev_params = (W_pre, W_pre, b_pre, b_pre, "linear", None)
    params = update_opt_params(prev_params, network_param, (L_hat, U_hat))
    W_L, W_U, b_L, b_U, _, _ = params
    assert torch.all(W_L - W_U == 0)
    L_lin, _ = fast_layer_matrix_form((W_L, b_L), L_0, U_0, activation=None)
    _, U_lin = fast_layer_matrix_form((W_U, b_U), L_0, U_0, activation=None)
    L_lin, U_lin = F.relu(L_lin), F.relu(U_lin)
    L_ibp, U_ibp = fast_layer_matrix_form((W_pre, b_pre), L_ibp, U_ibp, activation=nn.ReLU())
    #L_ibp, U_ibp = fast_layer_matrix_form((W_pre, b_pre), L_ibp, U_ibp, activation=None)
    tol = 1e-7
    print(np.all(L_ibp.numpy() - L_lin.numpy() < tol))
    print(np.all(U_ibp.numpy() - U_lin.numpy() > -tol))



    # ### test 3: single layer ibp is better than single layer linear appx for relu-convolution layer
    # from conv2d_mat import resolve_padded_dims, conv2d_circulant_like_multi_ch
    # from fast_appx import fast_conv2D_layer_matrix_form
    #
    # np.random.seed(10)
    # n_in, n_out, k_h, k_w = 1, 1, 2, 2
    # i_h, i_w = 4, 4
    # stride, padding = 1, 0
    # o_h, o_w = resolve_padded_dims(i_h, i_w, k_h, k_w, stride, padding)
    # W, b = torch.tensor(np.random.randn(n_out, n_in, k_h, k_w)).float(), torch.tensor(np.random.randn(n_out)).float()
    # network_param = ((W, b), "conv2d", "relu", (stride, padding))
    #
    # # init random hyper-rectangular bounds after layer l-1
    # L_0 = np.random.randn(n_in, i_h, i_w)
    # eps = 0.1
    # U_0 = L_0 + eps
    # L_0, U_0 = torch.tensor(L_0).float(), torch.tensor(U_0).float()
    # L_hat, U_hat = fast_conv2D_layer_matrix_form((W, b), (stride, padding),
    #                                               L_0, U_0, activation=None) # pre-act ibp bounds
    # L_ibp, U_ibp = F.relu(L_hat), F.relu(U_hat) # ibp for layer l
    #
    # # identity layer at l+1 (no effect)
    # d_out = n_out * o_h * o_w
    # prev_params = (torch.eye(d_out, d_out), torch.eye(d_out, d_out),
    #                torch.zeros(d_out), torch.zeros(d_out), "linear", None)
    # params = update_opt_params(prev_params, network_param, (L_hat, U_hat))
    # W_L, W_U, b_L, b_U, lin_op_type, _ = params
    # assert np.all(W_L.numpy() - W_U.numpy() == 0)
    #
    # if lin_op_type == "conv2d":
    #     W_L, W_U = W_L.view(W_L.size(0), -1), W_U.view(W_U.size(0), -1)
    #     L_0, U_0 = L_0.view(-1), U_0.view(-1)
    #
    # L_lin, _ = fast_layer_matrix_form((W_L, b_L), L_0, U_0, activation=None)
    # _, U_lin = fast_layer_matrix_form((W_U, b_U), L_0, U_0, activation=None)
    #
    # L_lin, U_lin = L_lin.view(L_hat.size()), U_lin.view(U_hat.size())
    # print(torch.min(L_ibp - L_lin))
    # print(torch.max(U_ibp-U_lin))
    # tol = 1e-05
    # assert torch.min(L_ibp-L_lin) >= -tol and torch.max(U_ibp-U_lin) <= tol



    # ### test 5: single layer linear appx for relu-convolution layer: equivalence between implicit and explicit matrix representation
    # W = W.reshape(n_in, k_h, k_w)
    # W_conv = conv2d_circulant_like_multi_ch(W, (i_h, i_w), stride=1, return_ls=False)
    # o_h, o_w = resolve_padded_dims(i_h, i_w, k_h, k_w, stride, padding)
    # assert n_out*o_h*o_w == W_conv.shape[0]
    # b_conv = conv2d_bias(b, o_h, o_w)
    # W_conv, b_conv = torch.tensor(W_conv).float(), torch.tensor(b_conv).float()
    # network_param = ((W_conv, b_conv), "linear", "relu", (stride, padding))
    # params = update_opt_params(prev_params, network_param, (L_hat.view(-1), U_hat.view(-1)))
    # W_L, W_U, b_L, b_U, lin_op_type, _ = params
    # assert np.all(W_L.numpy() - W_U.numpy() == 0)
    #
    # # if lin_op_type == "conv2d":
    # #     W_L, W_U = W_L.view(W_L.size(0), -1), W_U.view(W_U.size(0), -1)
    # #     L_0, U_0 = L_0.view(-1), U_0.view(-1)
    #
    # L_lin_ex, _ = fast_layer_matrix_form((W_L, b_L), L_0.view(-1), U_0.view(-1), activation=None)
    # _, U_lin_ex = fast_layer_matrix_form((W_U, b_U), L_0.view(-1), U_0.view(-1), activation=None)
    #
    # L_lin_ex, U_lin_ex = L_lin_ex.view(L_hat.size()), U_lin_ex.view(U_hat.size())
    # print(torch.min(L_lin_ex - L_lin))
    # print(torch.max(L_lin_ex - U_lin))
    #
    # assert torch.min(L_lin_ex - L_lin)==0 and torch.max(L_lin_ex - U_lin)==0





