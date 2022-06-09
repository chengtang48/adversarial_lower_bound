import numpy as np
from exact_solve import *


def create_init(x_0, eps):
    d = len(x_0)
    b_0_u = eps * np.ones(d, ) + x_0
    A_0_u = np.eye(d)
    A_0_l = -1 * A_0_u
    b_0_l = eps * np.ones(d, ) - x_0
    ###
    A_0 = np.concatenate((A_0_u, A_0_l), axis=0)
    b_0 = np.concatenate((b_0_u, b_0_l), axis=0).reshape(-1, 1)

    V = Node()
    pos_ind = range(d)
    neg_ind = range(d, 2 * d)
    extremes_init = []
    gen_V(d, V)

    for v in gen_path(V):
        ind = []
        for i in range(d):
            if v[i] == "0":
                ind.append(neg_ind[i])
            else:
                ind.append(pos_ind[i])
        print(v, ind)
        A_ij = A_0[ind, :]
        # print(A_ij)
        b_ij = np.concatenate([b_0[i] for i in ind], axis=0)
        # print(b_ij)
        v = np.linalg.inv(A_ij).dot(b_ij)
        # print(v)
        extremes_init.append(v)

    return A_0, b_0, extremes_init


def test_parse_res(c_obj, const, opt_res, final_res):
    # parse result from lp solver
    assert opt_res.status != 3
    if opt_res.status == 0:
        final_res["successes"].append(opt_res.x)
        curr_val = c_obj.dot(opt_res.x) + const
        if final_res["best_val_found"] is None:
            final_res["best_val_found"] = (opt_res.x, curr_val)
        elif curr_val < final_res["best_val_found"][1]:
            final_res["best_val_found"] = (opt_res.x, curr_val)
    elif opt_res.status == 2:
        final_res["num_infeasible"] += 1



def test_network_robustness(hidden_network_params, x_0, c_obj, eps):
    X_0 = [list()]
    Theta_0 = [(np.eye(len(x_0)), np.zeros(len(x_0)))]
    X, Theta, _ = forward(hidden_network_params, X_0, Theta_0, save_inter=False)
    ####
    A_0, b_0, extremes_init = create_init(x_0, eps)
    extremes_init = []
    print("vertices in initial polytope: ", extremes_init)
    ####
    final_res = {"best_val_found": None, "successes": [], "num_infeasible": 0}
    print(len(X), len(Theta))
    for (X_i, Theta_i) in zip(X, Theta):
        res, c_mod, const = solve_primal_lp(c_obj, A_0, b_0, X_i, Theta_i)
        test_parse_res(c_mod, const, res, final_res)
    return final_res, hidden_network_params



######
def run_example(c_obj, network_params, pt):
    # TODO: replace with network model forward
    x = pt
    for (W, b) in network_params:
        W_x = W.dot(x).reshape(b.shape)
        y = (W_x + b).reshape(-1)
        x = np.zeros(len(y))
        x[y>0] = y[y>0]
        print(W, b, x)
    return x, c_obj.dot(x)