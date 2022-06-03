import numpy as np
from main import Node, gen_V, gen_path, forward, solve_primal_lp

#### test functions
def create_init(x_0, eps):
    d = len(x_0)
    b_0_u = eps * np.ones(d, ) + x_0
    A_0_u = np.eye(d)
    A_0_l = -1 * A_0_u
    b_0_l = eps * np.ones(d, ) - x_0
    ###
    A_0 = np.concatenate((A_0_u, A_0_l), axis=0)
    b_0 = np.concatenate((b_0_u, b_0_l), axis=0).reshape(-1, 1)

    # print(A_0, b_0)

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


def run_example(c_obj, network_params, pt):
    x = pt
    for (W, b) in network_params:
        W_x = W.dot(x).reshape(b.shape)
        y = (W_x + b).reshape(-1)
        x = np.zeros(len(y))
        x[y > 0] = y[y > 0]
        print(W, b, x)
    return x, c_obj.dot(x)

############ test with toy networks
def gt_single_neuron(x_0, eps, w, b):
    bounds = []
    for i in range(len(x_0)):
        l_i = x_0[i] - eps
        u_i = x_0[i] + eps
        if w[i] >= 0:
            bounds.append((w[i] * l_i, w[i] * u_i))
        else:
            bounds.append((w[i] * u_i, w[i] * l_i))
    lo, hi = 0, 0
    for (l, u) in bounds:
        lo += l
        hi += u
    return max(lo + b, 0)


def test_single_neuron(x_0, eps):
    # same as a simple linear classifier
    d = len(x_0)
    n_neuron = 1
    W, b = 2 * (np.random.rand(n_neuron, d) - 0.5), 2 * (np.random.rand(n_neuron, 1) - 0.5)
    opt = gt_single_neuron(x_0, eps, W.reshape(-1), b)
    c_obj = np.array([3])
    # print("W: {}, b: {}, c: {}, opt: {}".format(W, b, c_obj, opt))
    hidden_network_params = [(W, b)]
    X_0 = [list()]
    Theta_0 = [(np.eye(d), np.zeros(d))]
    X, Theta, _ = forward(hidden_network_params, X_0, Theta_0)
    ##
    # print(X, Theta)
    ##
    A_0, b_0 = create_init(x_0, eps)
    final_res = {"best_val_found": None, "successes": [], "num_infeasible": 0}
    for (X_i, Theta_i) in zip(X, Theta):
        res, c_mod, const = solve_primal_lp(c_obj, A_0, b_0, X_i, Theta_i)
        test_parse_res(c_mod, const, res, final_res)
    return final_res


def test_two_neurons(x_0, c_obj, eps):
    # compute optimal value (check if it's consistent with the graphed value)
    d = len(x_0)
    n_neuron = 2
    W_nn, b_nn = 5 * (np.random.rand(n_neuron, len(x_0)) - 0.5), 2.5 * (np.random.rand(n_neuron, 1) - 0.5)
    print("W: {}, b: {}, c: {}".format(W_nn, b_nn, c_obj))
    hidden_network_params = [(W_nn, b_nn)]  # 1-layer-2-neurons
    X_0 = [list()]
    Theta_0 = [(np.eye(len(x_0)), np.zeros(len(x_0)))]
    X, Theta, _ = forward(hidden_network_params, X_0, Theta_0, save_inter=False)
    ####
    A_0, b_0 = create_init(x_0, eps)
    extremes_init = []
    for (i, j) in [(2, 1), (0, 1), (0, 3), (2, 3)]:
        A_ij = A_0[(i, j), :]
        # print(A_ij)
        b_ij = np.concatenate((b_0[i], b_0[j]), axis=0)
        # print(b_ij)
        v = np.linalg.inv(A_ij).dot(b_ij)
        # print(v)
        extremes_init.append(v)
    print("vertices in initial polytope: ", extremes_init)
    ####
    final_res = {"best_val_found": None, "successes": [], "num_infeasible": 0}
    for (X_i, Theta_i) in zip(X, Theta):
        res, c_mod, const = solve_primal_lp(c_obj, A_0, b_0, X_i, Theta_i)
        test_parse_res(c_mod, const, res, final_res)
    return final_res


##########
def test_2_2_layers(x_0, c_obj, eps):
    # A two-layer network
    n_neuron = 2
    L = 2
    hidden_network_params = []
    for l in range(L):
        np.random.seed(l + 5)
        W_nn, b_nn = 5 * (np.random.rand(n_neuron, len(x_0)) - 0.5), 2.5 * (np.random.rand(n_neuron, 1) - 0.5)
        hidden_network_params.append((W_nn, b_nn))
        print("Layer: {}, W: {}, b: {}".format(l, W_nn, b_nn))
    X_0 = [list()]
    Theta_0 = [(np.eye(len(x_0)), np.zeros(len(x_0)))]
    X, Theta, _ = forward(hidden_network_params, X_0, Theta_0, save_inter=False)
    ####
    A_0, b_0 = create_init(x_0, eps)
    extremes_init = []
    for (i, j) in [(2, 1), (0, 1), (0, 3), (2, 3)]:
        A_ij = A_0[(i, j), :]
        # print(A_ij)
        b_ij = np.concatenate((b_0[i], b_0[j]), axis=0)
        # print(b_ij)
        v = np.linalg.inv(A_ij).dot(b_ij)
        # print(v)
        extremes_init.append(v)
    print("vertices in initial polytope: ", extremes_init)
    ####
    final_res = {"best_val_found": None, "successes": [], "num_infeasible": 0}
    print(len(X), len(Theta))
    for (X_i, Theta_i) in zip(X, Theta):
        res, c_mod, const = solve_primal_lp(c_obj, A_0, b_0, X_i, Theta_i)
        test_parse_res(c_mod, const, res, final_res)
    return final_res, hidden_network_params


def test_3_2_layers(x_0, c_obj, eps):
    # one layer with 3 neurons, another with 2 layers
    n_neurons = [3, 2]
    L = 2
    hidden_network_params = []
    for l in range(L):
        np.random.seed(l + 10)
        input_dim = len(x_0) if l == 0 else n_neurons[l - 1]
        W_nn, b_nn = 5 * (np.random.rand(n_neurons[l], input_dim) - 0.5), 2.5 * (np.random.rand(n_neurons[l], 1) - 0.5)
        hidden_network_params.append((W_nn, b_nn))
        print("Layer: {}, W: {}, b: {}".format(l, W_nn, b_nn))
    X_0 = [list()]
    Theta_0 = [(np.eye(len(x_0)), np.zeros(len(x_0)))]
    X, Theta, _ = forward(hidden_network_params, X_0, Theta_0, save_inter=False)
    ####
    A_0, b_0, extremes_init = create_init(x_0, eps)
    print("vertices in initial polytope: ", extremes_init)
    ####
    final_res = {"best_val_found": None, "successes": [], "num_infeasible": 0}
    print(len(X), len(Theta))
    for (X_i, Theta_i) in zip(X, Theta):
        res, c_mod, const = solve_primal_lp(c_obj, A_0, b_0, X_i, Theta_i)
        test_parse_res(c_mod, const, res, final_res)
    return final_res, hidden_network_params


def test_1_3_layers(x_0, c_obj, eps):
    # one layer with 1 neuron, another with 3 neurons
    n_neurons = [1, 3]
    L = 2
    hidden_network_params = []
    for l in range(L):
        np.random.seed(l + 3)
        input_dim = len(x_0) if l == 0 else n_neurons[l - 1]
        W_nn, b_nn = 5 * (np.random.rand(n_neurons[l], input_dim) - 0.5), 2.5 * (np.random.rand(n_neurons[l], 1) - 0.5)
        hidden_network_params.append((W_nn, b_nn))
        print("Layer: {}, W: {}, b: {}".format(l, W_nn, b_nn))
    X_0 = [list()]
    Theta_0 = [(np.eye(len(x_0)), np.zeros(len(x_0)))]
    X, Theta, _ = forward(hidden_network_params, X_0, Theta_0, save_inter=False)
    ####
    A_0, b_0, extremes_init = create_init(x_0, eps)
    extremes_init = []
    #     for (i, j) in [(2, 1), (0, 1), (0, 3), (2, 3)]:
    #         A_ij = A_0[(i, j), :]
    #         #print(A_ij)
    #         b_ij = np.concatenate((b_0[i], b_0[j]), axis=0)
    #         #print(b_ij)
    #         v = np.linalg.inv(A_ij).dot(b_ij)
    #         #print(v)
    #         extremes_init.append(v)
    print("vertices in initial polytope: ", extremes_init)
    ####
    final_res = {"best_val_found": None, "successes": [], "num_infeasible": 0}
    print(len(X), len(Theta))
    for (X_i, Theta_i) in zip(X, Theta):
        res, c_mod, const = solve_primal_lp(c_obj, A_0, b_0, X_i, Theta_i)
        test_parse_res(c_mod, const, res, final_res)
    return final_res, hidden_network_params

if __name__ == "__main__":
    # TODO: add tests here
    # test single neuron
    # test two neurons
    # test two-layer toy networks
    pass
