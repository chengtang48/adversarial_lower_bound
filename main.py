import numpy as np
import argparse
from exact_solve import *
from ReLU_networks import classificationmodel

import torchvision.datasets as dts
import torchvision.transforms as trnsfrms
import torch


def create_init(x_0, eps, find_extremes_init=False):
    d = len(x_0)
    b_0_u = eps * np.ones(d, ) + x_0
    A_0_u = np.eye(d)
    A_0_l = np.diag(-1*np.ones(d)) #-1 * A_0_u
    b_0_l = eps * np.ones(d, ) - x_0
    #assert np.all(x_0 <= b_0_u)
    #assert np.all(-x_0 <= b_0_l)
    ###
    A_0 = np.concatenate((A_0_u, A_0_l), axis=0)
    #b_0 = np.concatenate((b_0_u, b_0_l), axis=0).reshape(-1, 1)
    b_0 = np.concatenate((b_0_u, b_0_l), axis=0)

    V = Node()
    pos_ind = range(d)
    neg_ind = range(d, 2 * d)
    extremes_init = []

    # if find_extremes_init:
    #     gen_V(d, V)
    #     for v in gen_path(V):
    #         ind = []
    #         for i in range(d):
    #             if v[i] == "0":
    #                 ind.append(neg_ind[i])
    #             else:
    #                 ind.append(pos_ind[i])
    #         #print(v, ind)
    #         A_ij = A_0[ind, :]
    #         # print(A_ij)
    #         b_ij = np.concatenate([b_0[i] for i in ind], axis=0)
    #         # print(b_ij)
    #         v = np.linalg.inv(A_ij).dot(b_ij)
    #         # print(v)
    #         extremes_init.append(v)
    return A_0, b_0, extremes_init


def create_init_bounds(x_0, eps):
    u = [x_0[i]+eps for i in range(len(x_0))]
    l = [x_0[i]-eps for i in range(len(x_0))]
    return l, u


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


######################################

def find_adv_attack(hidden_network_params, x_0, c_obj, eps):
    ####
    A_0, b_0, _ = create_init(x_0, eps, find_extremes_init=False)
    L, U = create_init_bounds(x_0, eps)
    ####
    X_0 = [list()]
    Theta_0 = [(np.eye(len(x_0)), np.zeros(len(x_0)))]
    X, Theta, _ = forward(hidden_network_params, X_0, Theta_0, save_inter=False, x_0=x_0, A_0=A_0, b_0=b_0, L=L, U=U)
    print("Done with forward computation")
    extremes_init = []
    # print("vertices in initial polytope: ", extremes_init)
    ####
    final_res = {"best_val_found": None, "successes": [], "num_infeasible": 0}
    print(len(X), len(Theta))
    for (X_i, Theta_i) in zip(X, Theta):
        res, c_mod, const = solve_primal_lp(c_obj, A_0, b_0, L, U, X_i, Theta_i)
        test_parse_res(c_mod, const, res, final_res)
    return final_res, hidden_network_params


def find_c_obj(W, outp, select='random'):
    #outp = model(x_0).detach().squeeze()
    # print(outp)
    max_idx = np.argmax(outp)
    j = 0
    if select == 'random':
        j = np.random.randint(len(outp))
        while j == max_idx:
            j = np.random.rand(len(outp))
    elif select == 'second':
        j_1, j_2 = None, None
        if len(outp[:max_idx]) > 0:
            j_1 = np.argmax(outp[:max_idx])
        if len(outp[max_idx + 1:]) > 0:
            j_2 = max_idx + np.argmax(outp[max_idx+1:]) + 1
        if j_2 is None or (j_1 is not None and outp[j_1] > outp[j_2]):
            j = j_1
        else:
            j = j_2
    #print("initial difference: ", outp[max_idx] - outp[j])
    print("max index: {}, selected index: {}".format(max_idx, j))
    #return W[max_idx, :] - W[j, :]
    return W[max_idx, :]


######
# def test_adv_example(c_obj, network_params, pt):
#     # TODO: replace with network model forward
#     x = pt
#     for (W, b) in network_params:
#         W_x = W.dot(x).reshape(b.shape)
#         y = (W_x + b).reshape(-1)
#         x = np.zeros(len(y))
#         x[y>0] = y[y>0]
#         print(W, b, x)
#     return x, c_obj.dot(x)


def run_adv_example(test_dataloader, model, hidden_network_params, select='exhaustive', num_runs=10, eps=0.1):
    total_runs = 0
    total_success = 0
    total_time = 0
    for images, labels in test_dataloader:
        preds = model(images).detach()
        preds_id = np.argmax(preds, axis=1)
        for idx, (pred_id, label) in enumerate(zip(preds_id, labels)):
            if pred_id == label:
                total_runs += 1
                outp = preds[idx].squeeze()
                print("original out pred {}".format(outp))
                if select == 'exhaustive':
                    max_idx = np.argmax(outp)
                    for j in range(len(W_L)):
                        if j != max_idx:
                            c_obj = W_L[max_idx, :] - W_L[j, :]
                            x_0 = images[idx].detach().numpy().reshape(-1, )
                            final_res, hidden_network_params = find_adv_attack(hidden_network_params, x_0, c_obj, eps)
                            if final_res['best_val_found'] is None:
                                continue
                            pt = final_res['best_val_found'][0]
                            adv_pred = model(torch.from_numpy(pt).float()).detach()
                            print("original: {} perturbed: {} label: {}".format(pred_id, adv_pred, label))
                            print("distance to x_0: {}".format(max(abs((pt - x_0)))))
                            if np.argmax(adv_pred, axis=1) != pred_id:
                                total_success += 1
                                break
                else:
                    c_obj = find_c_obj(W_L, outp, select=select)
                    x_0 = images[idx].detach().numpy().reshape(-1,)
                    final_res, hidden_network_params = find_adv_attack(hidden_network_params, x_0, c_obj, eps)
                    if final_res['best_val_found'] is None:
                        continue
                    pt = final_res['best_val_found'][0]
                    adv_pred = model(torch.from_numpy(pt).float()).detach()
                    print("original: {} perturbed: {} label: {}".format(pred_id, adv_pred, label))
                    print("distance to x_0: {}".format(max(abs((pt-x_0)))))
                    if np.argmax(adv_pred, axis=1) != pred_id:
                        total_success += 1
            if total_runs >= num_runs:
                break
        if total_runs >= num_runs:
            break

    print("Number of runs: {} Number of successful attacks: {}".format(total_runs, total_success))


#### interface with networks
def get_network_params(model):
    all_network_params = dict()
    for idx, (name, param) in enumerate(model.named_parameters()):
        if name[:8] not in all_network_params:
            all_network_params[name[:8]] = [param.data.detach().numpy()]
        else:
            all_network_params[name[:8]].append(param.data.detach().numpy())
    network_params = []
    for key, param in all_network_params.items():
        if len(param) == 2:
            W, b = param[0], param[1]
            network_params.append((W, b))
            #print(W.shape, b.shape)
        else:
            W = param[0]
            #network_params.append(W)
            #print(W.shape)
    return network_params, W

####




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    args = parser.parse_args()

    model = classificationmodel(network_params=(6, 6, 10))  # add any non-default init params
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    #network_params = get_network_params(model)
    #hidden_network_params, W_L = network_params[:-1], network_params[-1]
    hidden_network_params, W_L = get_network_params(model)

    # x_0 = torch.randn((28*28,))
    trnsform = trnsfrms.Compose([trnsfrms.ToTensor(), trnsfrms.Normalize((0.7,), (0.7,))])
    #mnist_testset = dts.MNIST(root='./data', train=False, download=False, transform=trnsform)
    mnist_trainset = dts.MNIST(root='./data', train=False, download=False, transform=trnsform)
    trainldr = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)
    #testldr = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=False)

    #print(len(hidden_network_params), W_L.shape)
    #for W, b in hidden_network_params:
    #    print(W.shape)
    run_adv_example(trainldr, model, hidden_network_params, select="random", num_runs=1, eps=0.1)