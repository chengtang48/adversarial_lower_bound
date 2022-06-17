import numpy as np
np.random.seed(0)
import copy
from scipy.optimize import linprog

#### iterator over orthants in R^d
class Node:
    def __init__(self, bit=""):
        self.left = None
        self.right = None
        self.bit = bit


def gen_V(d, V):
    # each path corresponds to an orthant
    if d == 0:
        return
    V.left = Node(bit="0")
    V.right = Node(bit="1")
    gen_V(d-1, V.left)
    gen_V(d-1, V.right)


def gen_path(ptr, seq=""):
    # depth d
    seq = seq + ptr.bit
    if ptr.left is None:
        yield seq
    else:
        yield from gen_path(ptr.left, seq=seq)
    if ptr.right is not None:
        yield from gen_path(ptr.right, seq=seq)


##### forward exact computation of embedding polytopes
def update_ineq(P, T, W, b, B, d, l):
    # add new constraint Ax <= b to P
    # P: list of l dictionaries
    # TODO: test CORRECTNESS
    # IT MAYBE more efficient to determine feasibility here
    # print("T: ", T)
    #print(T.shape, W.shape, B.shape)
    A_1 = -1 * np.matmul(T, np.matmul(W, B))
    W_d = W.dot(d).reshape(b.shape) # !!! avoid broadcast in later addition
    #print(S.shape, W.shape, d.shape, b.shape)
    b_1 = T.dot(W_d + b)
    #print(W.dot(d).shape, b_1.shape)
    #A_2 = np.matmul(W, B) + A_1
    #b_2 = -1 * (W_d + b) + b_1
    #P.append({"I_{}_1".format(l+1):(A_1, b_1), "I_{}_2".format(l+1):(A_2, b_2)})
    P.append((A_1, b_1))
    return P


def layer(network_param, X, Theta, l, x_0=None, A_0=None, b_0=None, L=None, U=None):
    X_new = []
    Theta_new = []
    W, b = network_param
    root = Node()
    gen_V(W.shape[0], root)
    i, j = 0, 0
    n_skipped = 0
    for X_i, Theta_i in zip(X, Theta):
        #print("poly_{}".format(i))
        P = X_i
        B, d = Theta_i
        # check if current polytope is feasible
        if l == 0:
            x_0 = x_0
        else:
            x_0 = None
        if is_infeasible(A_0, b_0, L, U, P, x_0=x_0):
            n_skipped += 1
            continue
        for v in gen_path(root):
            #print("orthant_{}".format(j))
            S_v = np.diag([float(int(vj=="1")) for vj in v])
            T_v = S_v - np.diag([float(int(vj=="0")) for vj in v])
            # TODO: save some matmul here
            P_ = update_ineq(copy.deepcopy(P), T_v, W, b, B, d, l) # update P by appending new inequalities
            #print("polytope: ", P_)
            B_ = np.matmul(S_v, np.matmul(W, B))
            W_d = W.dot(d).reshape(b.shape) ## !!! AVOID broadcast
            # d_ = S_v.dot(W.dot(d)+b)
            d_ = S_v.dot(W_d + b)
            X_new.append(P_)
            Theta_new.append((B_, d_))
            j += 1
        i += 1
        #print(X_new)
    return X_new, Theta_new, n_skipped


def forward(network_params, X, Theta, save_inter=False, x_0=None, A_0=None, b_0=None, L=None, U=None):
    # TODO: add timer
    all_X_Theta = []

    for i in range(len(network_params)):
        print("Forward computation at layer {}".format(i+1))
        X, Theta, n_skipped = layer(network_params[i], X, Theta, i, x_0, A_0, b_0, L, U)
        print("Done..skipped {} polytopes".format(n_skipped))
        if save_inter:
            all_X_Theta.append((X, Theta))
    return X, Theta, all_X_Theta if len(all_X_Theta) else None


#### LP solve
def is_infeasible(A_0, b_0, L, U, P, x_0=None):
    #A, b = A_0, b_0
    A, b = None, None
    for l in range(len(P)):
        A_l, b_l = P[l]
        if A is None:
            A, b = A_l, b_l
        else:
            # print(A_l.shape, b_l.shape)
            A = np.concatenate((A, A_l), axis=0)
            b = np.concatenate((b, b_l), axis=0)
    c_const = np.zeros(A_0.shape[1])
    res = linprog(c_const, A_ub=A, b_ub=b,
                  A_eq=None, b_eq=None, bounds=list(zip(L, U)),
                  method='interior-point')
    return res.status == 2



# def solve_primal_lp(c, A_0, b_0, P, theta):
#     # Input: P with multiple ineq A_i x <= b_i,
#     # test solver: scipy linprog
#     A, b = A_0, b_0  # INIT CONSTRAINT
#     #print("init: ", A_0.shape, b_0.shape)
#     for l in range(len(P)):
#         A_l, b_l = P[l]
#         #print(A_l.shape, b_l.shape)
#         A = np.concatenate((A, A_l), axis=0)
#         b = np.concatenate((b, b_l), axis=0)
#
#     # print(A)
#     # print(b)
#
#     # transform objective
#     B, d = theta
#     const = c.dot(d)
#     c_mod = c.dot(B)
#
#     # solve (note: this is solving a minimization problem)
#     res = linprog(c_mod, A_ub=A, b_ub=b,
#                   A_eq=None, b_eq=None, bounds=None,
#                   method='interior-point',
#                   callback=None, options=None, x0=None)
#
#     return res, c_mod, const

def solve_primal_lp(c, A_0, b_0, L, U, P, theta):
    # Input: P with multiple ineq A_i x <= b_i,
    # test solver: scipy linprog
    #A, b = A_0, b_0  # INIT CONSTRAINT
    A, b = None, None
    #print("init: ", A_0.shape, b_0.shape)
    for l in range(len(P)):
        A_l, b_l = P[l]
        if A is None:
            A, b = A_l, b_l
        else:
            #print(A_l.shape, b_l.shape)
            A = np.concatenate((A, A_l), axis=0)
            b = np.concatenate((b, b_l), axis=0)

    # print(A)
    # print(b)

    # transform objective
    B, d = theta
    const = c.dot(d)
    c_mod = c.dot(B)

    # solve (note: this is solving a minimization problem)
    res = linprog(c_mod, A_ub=A, b_ub=b,
                  A_eq=None, b_eq=None, bounds=list(zip(L, U)),
                  method='interior-point',
                  callback=None, options=None, x0=None)

    return res, c_mod, const



