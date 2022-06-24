### this approx relies on property of ReLU
import numpy as np
import torch
import torch.nn.functional as F


def get_extreme(w, L, U, direction="max"):
    res = 0
    if direction == 'max':
        for i in range(len(L)):
            if w[i] >= 0:
                res += w[i]*U[i]
            else:
                res += w[i]*L[i]
    elif direction == 'min':
        for i in range(len(L)):
            if w[i] >= 0:
                res += w[i]*L[i]
            else:
                res += w[i]*U[i]
    return res


def get_min(w, L, U):
    res = 0
    x_star = np.zeros(len(L))
    for i in range(len(L)):
        if w[i] >= 0:
            res += w[i] * L[i]
            x_star[i] = L[i]
        else:
            res += w[i] * U[i]
            x_star[i] = U[i]
    return res, x_star


def get_R(w, b, L, U):
    u = get_extreme(w, L, U, direction="max")
    l = get_extreme(w, L, U, direction="min")
    return max(0, l+b), max(0, u+b)


def fast_layer(network_param, L, U):
    W, b = network_param
    L_new, U_new = list(), list()
    for i in range(len(b)):
        l, u = get_R(W[i, :], b[i], L, U)
        L_new.append(l)
        U_new.append(u)
    return L_new, U_new


def fast_layer_matrix_form(network_param, L, U):
    """
    matrix form of fast-layer update, via pytorch
    :param network_param: W, b
    :param L: tensorized lower bound vector
    :param U: tensorized upper bound vector
    :return: L_new, U_new
    """
    W, b = network_param
    o0 = torch.matmul(W, U)
    o1 = torch.matmul(F.relu(W), L - U)
    #L_new = F.relu(F.linear(U, W, bias=b+o1))
    L_new =  F.relu(torch.add(b, o0 + o1))
    o2 = torch.matmul(F.relu(-1*W), U-L)
    #U_new = F.relu(F.linear(U, W, bias=b+o2))
    U_new = F.relu(torch.add(b, o0 + o2))
    return L_new, U_new


def fast_conv2D_layer_matrix_form(network_param, conv_param, L, U):
    """
        approximates outputs from conv2D layer, implemented directly via pytorch
        :param network_param: W, b
        :param L: tensorized lower bound matrix, shape=(d, d)
        :param U: tensorized upper bound matrix, shape=(d, d)
        :return: L_new, U_new
    """
    W_k, b_k = network_param
    stride, padding, *params = conv_param
    o0 = F.conv2d(U, W_k, bias=b_k, stride=stride, padding=padding, *params)
    o1 = F.conv2d(L-U, F.relu(W_k), bias=None, stride=stride, padding=padding, *params)
    L_new = F.relu(o1 + o0)

    o2 = F.conv2d(U-L, F.relu(-1*W_k), bias=None, stride=stride, padding=padding, *params)
    #U_new = F.relu(F.conv2d(U, W_k, bias=b_k+o2, stride=stride, padding=padding, *params))
    U_new = F.relu(o2 + o0)
    return L_new, U_new


def fast_pool2D_layer_matrix_form(kernel_size, conv_param, L, U, op='max'):
    """
    # TODO: need to test correctness for max-pool
    avg or max pool of input range <=> avg or max pool of L/U
    :param kernel_size: pooling size
    :param conv_param: stride, padding, and other params
    :param L:
    :param U:
    :param op: "max" for max-pooling; "avg" for average pooling
    :return: L_new, U_new
    """
    assert op in ['max', 'avg']
    stride, padding, *params = conv_param
    if op == 'max':
        L_new = F.max_pool2d(L, kernel_size, stride=stride, padding=padding, *params)
        U_new = F.max_pool2d(U, kernel_size, stride=stride, padding=padding, *params)
    elif op == 'avg':
        L_new = F.avg_pool2d(L, kernel_size, stride=stride, padding=padding, *params)
        U_new = F.avg_pool2d(U, kernel_size, stride=stride, padding=padding, *params)
    return L_new, U_new



##### callable by find_adv_attack
def fast_forward(hidden_network_params, L_0, U_0):
    L, U = L_0, U_0
    for network_param in hidden_network_params:
        L, U = fast_layer(network_param, L, U)
    return L, U


if __name__ == '__main__':
    # test linear layer matrix form
    np.random.seed(0)

    W = np.random.rand(10, 100)
    b = np.random.rand(10)
    eps = 0.1
    L = np.random.ones(100)
    U = L + eps
    L_a, U_a  = fast_layer((W, b), L, U)
    L_m, U_m = fast_layer_matrix_form((torch.tensor(W), torch.tensor(b)), torch.tensor(L), torch.tensor(U))
    print("Test 1: |DL|/|L|: {}; |DU|/|U|: {}".format(np.linalg.norm(np.array(L_a)-L_m.detach().numpy())/np.linalg.norm(np.array(L_a)),
                                     np.linalg.norm(np.array(U_a) - U_m.detach().numpy())/np.linalg.norm(np.array(U_a))))

    ### tests for conv2D layer: relu/negation op on kernel <=> relu/negation op on transformation matrix
    from conv2d_mat import conv2d_circulant_like_multi_ch

    # single-output multiple-input channels
    W_k = np.random.randn(3, 3, 3) #n_in-h_k-w_k
    # print(W_k)
    L = np.random.randn(3, 100, 100) #n_in-h_i-w_i
    stride, padding = 1, 0
    o_h, o_w = (L.shape[-2] - W_k.shape[-2]) // stride + 1, (L.shape[-1] - W_k.shape[-1]) // stride + 1
    b_k = np.zeros(1)
    eps = 0.1
    U = L + eps
    T_W_k = conv2d_circulant_like_multi_ch(W_k, (L.shape[-2], L.shape[-1]), stride=stride, return_ls=False)

    L_m, U_m = fast_layer_matrix_form((torch.tensor(T_W_k).reshape(1, *T_W_k.shape),
                                       torch.tensor(b_k).reshape(1, *b_k.shape)),
                                      torch.tensor(L.reshape(-1)), torch.tensor(U.reshape(-1)))
    L_m = L_m.reshape(o_h, o_w)
    U_m = U_m.reshape(o_h, o_w)

    W_k_c = torch.tensor(W_k.reshape(1, *W_k.shape))
    b_k_c = torch.tensor(np.array([0])) # each output channel has only one bias

    L_c, U_c = fast_conv2D_layer_matrix_form((W_k_c, b_k_c), (stride, padding), torch.tensor(L), torch.tensor(U))

    print(np.linalg.norm(L_m))
    print("Test 2: |DL|/|L|: {}; |DU|/|U|: {}".format(np.linalg.norm(L_m-L_c.detach().numpy())/np.linalg.norm(L_m),
    np.linalg.norm(U_m-U_c.detach().numpy())/np.linalg.norm(U_m)))

    # multi-output multi-input channels with stride=2, padding=1 and nonzero bias
    W_k = np.random.randn(6, 3, 4, 4)
    L = np.random.randn(3, 100, 100)
    eps = 0.05
    U = L + eps
    stride, padding = 2, 1
    L_m, U_m = L, U
    if padding > 0:
        # pad each input channel
        inpL, inpU = [], []
        for i in range(L_m.shape[0]):
            inpL.append(np.pad(L_m[i, :, :], padding))
            inpU.append(np.pad(U_m[i, :, :], padding))
        L_m = np.stack(inpL, axis=0)
        U_m = np.stack(inpU, axis=0)


    o_h, o_w = (L_m.shape[-2] - W_k.shape[-2]) // stride + 1, (L_m.shape[-1] - W_k.shape[-1]) // stride + 1
    b_k_m = np.random.randn(W_k.shape[0])
    T_W_k_m = []
    for i in range(W_k.shape[0]):
        T_W_k_m.append(conv2d_circulant_like_multi_ch(W_k[i], (L_m.shape[-2], L_m.shape[-1]), stride=stride, return_ls=False))
    T_W_k_m = np.stack(T_W_k_m, axis=0) # shape = (n_out, o_h*o_w, n_in*i_h*i_w)

    #print(T_W_k_m.shape, L_m.shape)
    L_m, U_m = fast_layer_matrix_form((torch.tensor(T_W_k_m), torch.tensor(b_k_m).reshape(-1, 1)),
                                       torch.tensor(L_m.reshape(-1)), torch.tensor(U_m.reshape(-1)))
    L_m = L_m.reshape(W_k.shape[0], o_h, o_w)
    U_m = U_m.reshape(W_k.shape[0], o_h, o_w)


    W_k_c = torch.tensor(W_k)
    b_k_c = torch.tensor(b_k_m) # each output channel has only one bias

    L_c, U_c = fast_conv2D_layer_matrix_form((W_k_c, b_k_c), (stride, padding), torch.tensor(L), torch.tensor(U))

    print("Test 3: |DL|/|L|: {}; |DU|/|U|: {}".format(np.linalg.norm(L_m-L_c.detach().numpy())/np.linalg.norm(L_m),
    np.linalg.norm(U_m-U_c.detach().numpy())/np.linalg.norm(U_m)))

    ### test for average-pooling layer







