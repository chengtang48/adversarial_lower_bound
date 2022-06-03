import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



def graph_2d_polytope(A, b, center):
    extremes = []
    for (i, j) in [(2, 1), (0, 1), (0, 3), (2, 3)]:
        A_ij = A[(i, j), :]
        # print(A_ij)
        b_ij = np.concatenate((b[i], b[j]), axis=0)
        # print(b_ij)
        v = np.linalg.inv(A_ij).dot(b_ij)
        # print(v)
        extremes.append(v)
    # print(extremes)
    ax = plt.gca()
    p = plt.Polygon(extremes)
    p.set_fill(True)
    ax.set_xlim(center[0] - 0.2, center[0] + 0.2)
    ax.set_ylim(center[1] - 0.2, center[1] + 0.2)
    ax.add_patch(p)
    plt.show()


def graph_2d_clipped_polytope(extremes, p_list, theta_list, c_obj, last_layer=True):
    patches = []
    extremes_new = []
    for ext in extremes:
        # determine which quadrant ext belongs to
        matching_ind = -1
        for ind, (A, b) in enumerate(p_list):
            # A, b = A[4:, :], b[4:, :]
            y = A.dot(ext).reshape(b.shape)
            if np.all(y - b <= 0):
                matching_ind = ind
                break
        print(matching_ind)
        B, d = theta_list[matching_ind]
        # print(B.shape, d.shape, ext.shape, (B.dot(ext) + d).shape)
        B_ext = B.dot(ext).reshape(-1, 1)
        extremes_new.append((B_ext + d).reshape(-1, ))
    print(extremes_new)
    ###
    ax = plt.gca()
    # p = plt.Polygon(extremes_new)
    # p.set_fill(True)
    extremes_new = np.array(extremes_new)
    if extremes_new.shape[1] == 1:
        xs, ys = extremes_new[:, 0], np.zeros(len(extremes_new))
    else:
        xs, ys = extremes_new[:, 0], extremes_new[:, 1]

    plt.scatter(xs, ys)

    x_left, x_right = min(extremes_new[:, 0]), max(extremes_new[:, 0])
    if extremes_new.shape[1] == 1:
        y_down, y_up = 0, 0
    else:
        y_down, y_up = min(extremes_new[:, 1]), max(extremes_new[:, 1])

    x = np.linspace(x_left - 0.1, x_right + 0.1, 20)
    # c_obj[0]*x + c_obj[1]*y=c_obj.dot(ext)
    if last_layer:
        for ext in extremes_new:
            plt.plot(x, (c_obj.dot(ext) - c_obj[0] * x) / c_obj[1])
        for pt in np.linspace((x_left, y_up), (x_right, y_down), 5):
            plt.plot(x, (c_obj.dot(pt) - c_obj[0] * x) / c_obj[1])
    ax.set_xlim(x_left - 0.1, x_right + 0.1)
    ax.set_ylim(y_down - 0.1, y_up + 0.1)
    # ax.add_patch(p)
    plt.show()
    return extremes_new


def graph_2d_preReLU_polytope(pre_extremes, network_param):
    patches = []
    extremes_new = []
    W, b = network_param
    for ext in pre_extremes:
        W_x = W.dot(ext).reshape(b.shape)
        x = (W_x + b).reshape(-1)
        extremes_new.append(x)
    print("preReLU extremes", extremes_new)
    ###
    ax = plt.gca()
    # p = plt.Polygon(extremes_new)
    extremes_new = np.array(extremes_new)
    if extremes_new.shape[1] == 1:
        xs, ys = extremes_new[:, 0], np.zeros(len(extremes_new))
    else:
        xs, ys = extremes_new[:, 0], extremes_new[:, 1]

    plt.scatter(xs, ys)

    x_left, x_right = min(extremes_new[:, 0]), max(extremes_new[:, 0])

    if extremes_new.shape[1] == 1:
        y_down, y_up = 0, 0
    else:
        y_down, y_up = min(extremes_new[:, 1]), max(extremes_new[:, 1])
    # plt.plot(np.linspace(x_left-0.1, x_right+0.1, 20), [0]*20, "b")
    # plt.plot([0]*20, np.linspace(y_down-0.1, y_up+0.1, 20), "r")
    # p.set_fill(True)

    ax.set_xlim(x_left - 0.1, x_right + 0.1)
    ax.set_ylim(y_down - 0.1, y_up + 0.1)
    # ax.add_patch(p)
    plt.show()


def graph_3d_polytope(extremes):
    print("extremes: ", extremes)
    fig = plt.figure()
    ax = Axes3D(fig)
    # fig.add_axes(ax, '3d')
    extremes = np.array(extremes)
    xs, ys, zs = extremes[:, 0], extremes[:, 1], extremes[:, 2]
    # ax.add_collection3d(Poly3DCollection(extremes))
    # extremes = np.array(extremes)
    ax.scatter3D(xs, ys, zs)
    ##
    # extremes = np.array(extremes)
    ax.axes.set_xlim3d(min(extremes[:, 0]) - 0.1, max(extremes[:, 0]) + 0.1)
    ax.axes.set_ylim3d(min(extremes[:, 1]) - 0.1, max(extremes[:, 1]) + 0.1)
    ax.axes.set_zlim3d(min(extremes[:, 2]) - 0.1, max(extremes[:, 2]) + 0.1)
    plt.show()


def graph_3d_preReLU_polytope(pre_extremes, network_param):
    patches = []
    extremes_new = []
    W, b = network_param
    for ext in pre_extremes:
        W_x = W.dot(ext).reshape(b.shape)
        x = (W_x + b).reshape(-1)
        extremes_new.append(x)
    #########################
    fig = plt.figure()
    ax = Axes3D(fig)
    extremes = np.array(extremes_new)
    xs, ys, zs = extremes[:, 0], extremes[:, 1], extremes[:, 2]
    ax.scatter3D(xs, ys, zs)
    ##
    ax.axes.set_xlim3d(min(extremes[:, 0]) - 0.1, max(extremes[:, 0]) + 0.1)
    ax.axes.set_ylim3d(min(extremes[:, 1]) - 0.1, max(extremes[:, 1]) + 0.1)
    ax.axes.set_zlim3d(min(extremes[:, 2]) - 0.1, max(extremes[:, 2]) + 0.1)

    # t = range(20)
    x_left, x_right = min(extremes[:, 0]), max(extremes[:, 0])
    y_down, y_up = min(extremes[:, 1]), max(extremes[:, 1])
    z_near, z_far = min(extremes[:, 2]), max(extremes[:, 2])

    ax.plot(np.linspace(x_left, x_right, 20), [0] * 20, [0] * 20)
    ax.plot([0] * 20, np.linspace(y_down, y_up, 20), [0] * 20)
    ax.plot([0] * 20, [0] * 20, np.linspace(z_near, z_far, 20))
    plt.show()


def graph_3d_clipped_polytope(extremes, p_list, theta_list, c_obj, last_layer=True):
    patches = []
    extremes_new = []
    for ext in extremes:
        # determine which quadrant ext belongs to
        matching_ind = -1
        for ind, (A, b) in enumerate(p_list):
            # A, b = A[4:, :], b[4:, :]
            y = A.dot(ext).reshape(b.shape)
            if np.all(y - b <= 0):
                matching_ind = ind
                break
        print(matching_ind)
        B, d = theta_list[matching_ind]
        # print(B.shape, d.shape, ext.shape, (B.dot(ext) + d).shape)
        B_ext = B.dot(ext).reshape(-1, 1)
        extremes_new.append((B_ext + d).reshape(-1, ))
    print(extremes_new)
    ###
    fig = plt.figure()
    ax = Axes3D(fig)
    extremes = np.array(extremes_new)
    xs, ys, zs = extremes[:, 0], extremes[:, 1], extremes[:, 2]
    ax.scatter3D(xs, ys, zs)
    ####
    ax.axes.set_xlim3d(min(extremes[:, 0]) - 0.1, max(extremes[:, 0]) + 0.1)
    ax.axes.set_ylim3d(min(extremes[:, 1]) - 0.1, max(extremes[:, 1]) + 0.1)
    ax.axes.set_zlim3d(min(extremes[:, 2]) - 0.1, max(extremes[:, 2]) + 0.1)
    ####
    x_left, x_right = min(extremes[:, 0]), max(extremes[:, 0])
    y_down, y_up = min(extremes[:, 1]), max(extremes[:, 1])
    z_near, z_far = min(extremes[:, 2]), max(extremes[:, 2])

    x = np.linspace(x_left - 0.1, x_right + 0.1, 20)
    y = np.linspace(y_down - 0.1, y_up + 0.1, 20)

    if last_layer:
        # c_obj[0]*x + c_obj[1]*y + c_obj[2]*z=c_obj.dot(ext)
        for ext in extremes:
            plt.plot(x, y, (c_obj.dot(ext) - c_obj[0] * x - c_obj[1] * y) / c_obj[2])

        for pt in np.linspace((x_left, y_down, z_near), (x_right, y_up, z_far), 5):
            plt.plot(x, y, (c_obj.dot(pt) - c_obj[0] * x - c_obj[2] * y) / c_obj[2])

    plt.show()

    return extremes_new

if __name__ == '__main__':
    # TODO: add viz tests here
    # test plotting 2D and 3D
    pass