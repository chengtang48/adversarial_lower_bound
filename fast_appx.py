### this approx relies on property of ReLU
import numpy as np

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


##### callable by find_adv_attack
def fast_forward(hidden_network_params, L_0, U_0):
    L, U = L_0, U_0
    for network_param in hidden_network_params:
        L, U = fast_layer(network_param, L, U)
    return L, U


