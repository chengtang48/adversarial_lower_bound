import numpy as np
import argparse

from exact_solve import *
from fast_appx import *
from ReLU_networks import FCModel

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


def find_appx_adv_attack(hidden_network_params, x_0, eps, W_last, max_idx):
    L_0, U_0 = create_init_bounds(x_0, eps)
    L, U = fast_forward(hidden_network_params, L_0, U_0)
    # exhaustive search
    x_sol = None
    for j in range(len(W_last)):
        if j == max_idx:
            continue
        c_obj = W_last[max_idx, :] - W_last[j, :]
        res, x_star = get_min(c_obj, L, U)
        if res < 0:
            x_sol = x_star
            break
    return x_sol


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
def run_benign_ex_adv_attack(test_dataloader, failed_ex, model, hidden_network_params, W_L, eps=0.1):
    good_ex = set(failed_ex)
    print("n_good_ex: ", len(good_ex))
    idx = 0
    n_success = 0
    for images, labels in test_dataloader:
        for image, label in zip(images, labels):
            if idx in good_ex:
                print("lower bounded example id: ", idx)
                pred = model(image).detach().squeeze()
                max_idx = np.argmax(pred)
                print("original out pred {}".format(pred))
                for j in range(len(W_L)):
                    if j != max_idx:
                        c_obj = W_L[max_idx, :] - W_L[j, :]
                        x_0 = image.detach().numpy().reshape(-1, )
                        final_res, hidden_network_params = find_adv_attack(hidden_network_params, x_0, c_obj, eps)
                        if final_res['best_val_found'] is None:
                            continue
                        pt = final_res['best_val_found'][0]
                        adv_pred = model(torch.from_numpy(pt).float()).detach()
                        print("adv out pred {}".format(adv_pred))
                        adv_max_idx = np.argmax(adv_pred)
                        print("original: {} perturbed: {} label: {}".format(max_idx, adv_max_idx, label))
                        print("distance to x_0: {}".format(max(abs((pt - x_0)))))
                        n_success += int(adv_max_idx != max_idx)
            idx += 1
    return n_success


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
                    print("SNR: {}".format(np.linalg.norm(x_0) / np.linalg.norm(pt-x_0)))
                    if np.argmax(adv_pred, axis=1) != pred_id:
                        total_success += 1
            if total_runs >= num_runs:
                break
        if total_runs >= num_runs:
            break

    print("Number of runs: {} Number of successful attacks: {}".format(total_runs, total_success))


def run_appx_adv_example(test_dataloader, model, hidden_network_params, W_last, num_runs=10, eps=0.1):
    total_runs = 0
    total_fails = 0
    total_time = 0
    failed_ex = []
    ex_id = 0
    for images, labels in test_dataloader:
        preds = model(images).detach()
        preds_id = np.argmax(preds, axis=1)
        for idx, (pred_id, label) in enumerate(zip(preds_id, labels)):
            if pred_id == label:
                total_runs += 1
                outp = preds[idx].squeeze()
                print("original out pred {}".format(outp))
                max_idx = np.argmax(outp)
                x_0 = images[idx].detach().numpy().reshape(-1, )
                x_sol = find_appx_adv_attack(hidden_network_params, x_0, eps, W_last, max_idx)
                if x_sol is not None:
                    # adv_pred = model(torch.from_numpy(x_sol).float()).detach()
                    adv_pred = W_last.dot(x_sol)
                    print("appx sol found!")
                    print("original: {} pseudo-perturbed: {} label: {}".format(pred_id, adv_pred, label))
                    #print("distance to x_0: {}".format(max(abs((x_sol - outp.numpy())))))
                else:
                    total_fails += 1
                    failed_ex.append(ex_id)
            ex_id += 1
            if total_runs >= num_runs:
                break
        if total_runs >= num_runs:
            break

    print("Number of runs: {} Number of lower bounded failure attacks: {}".format(total_runs, total_fails))
    return failed_ex



#### interface with networks
# def get_network_params(model):
#     all_network_params = dict()
#     for idx, (name, param) in enumerate(model.named_parameters()):
#         if name[:8] not in all_network_params:
#             all_network_params[name[:8]] = [param.data.detach().numpy()]
#         else:
#             all_network_params[name[:8]].append(param.data.detach().numpy())
#     network_params = []
#     for key, param in all_network_params.items():
#         if len(param) == 2:
#             W, b = param[0], param[1]
#             network_params.append((W, b))
#             #print(W.shape, b.shape)
#         else:
#             W = param[0]
#             #network_params.append(W)
#             #print(W.shape)
#     return network_params, W


def get_network_params(model):
    network_params, W_L = [], None
    model_params = list(model.parameters())
    for idx, param in enumerate(model_params):
        if idx == len(model_params)-1:
            W_L = param.data.detach().numpy()
        else:
            if idx % 2 == 0:
                param_data = [None, None]
                param_data[0] = param.data.detach().numpy()
            else:
                param_data[1] = param.data.detach().numpy()
                network_params.append(tuple(param_data))
    return network_params, W_L
####


if __name__ == '__main__':
    # TODO: adversarial training: apply fast appx to adversarial training (compare with naive Lipschitz bound)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config')
    parser.add_argument('--model_path')
    args = parser.parse_args()

    if args.model_config:
        param_str = args.model_config.split(",")
        d_in, network_param_str = int(param_str[0]), param_str[1:]
        network_params = []
        for p in network_param_str:
            network_params.append(int(p))
        fcmodel = FCModel(network_params=tuple(network_params), d_in=d_in)
    else:
        fcmodel = FCModel()

    fcmodel.load_state_dict(torch.load(args.model_path))
    fcmodel.eval()

    for param_tensor in fcmodel.state_dict():
        print(param_tensor, "\t", fcmodel.state_dict()[param_tensor].size())


    hidden_network_params, W_L = get_network_params(fcmodel)
    # for (W, b) in hidden_network_params:
    #     print(W.shape, b.shape)
    # print(W_L.shape)
    #

    trnsform = trnsfrms.Compose([trnsfrms.ToTensor(), trnsfrms.Normalize((0.7,), (0.7,))])
    mnist_testset = dts.MNIST(root='./data', train=False, download=False, transform=trnsform)
    #mnist_trainset = dts.MNIST(root='./data', train=False, download=False, transform=trnsform)
    #trainldr = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)
    testldr = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=False)

    #print(len(hidden_network_params), W_L.shape)
    #for W, b in hidden_network_params:
    #    print(W.shape)
    #run_adv_example(testldr, fcmodel, hidden_network_params, select="exhaustive", num_runs=100, eps=0.01)
    eps = 0.05
    failed_ex = run_appx_adv_example(testldr, fcmodel, hidden_network_params, W_L, num_runs=10, eps=eps)

    n_success = run_benign_ex_adv_attack(testldr, failed_ex, fcmodel, hidden_network_params, W_L, eps=eps)
    print("n_successes / n_lower_bounded_fails: {}/{}".format(n_success, len(failed_ex)))

