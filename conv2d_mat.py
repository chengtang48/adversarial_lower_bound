import numpy as np
import scipy.linalg as linalg
import torch

"""
Construct matrix representation of 2d-convolution operation (conv2d as implemented by pytorch)

code modified from
https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n
"""

def resolve_padded_dims(i_h, i_w, k_h, k_w, stride, padding):
    i_h, i_w = i_h + 2*padding, i_w + 2*padding
    return (i_h-k_h)//stride+1, (i_w-k_w)//stride+1


def conv2d_bias(b, o_h, o_w):
    """
    compute bias param in explicit affine representation of conv2d;
    :param b: bias param from a conv2d net; shape=(n_out,)
    :param o_h: output height of conv2d op
    :param o_w: output width of conv2d op
    :return: b_ex: shape=(n_out * o_h * o_w)
    """
    n_out = len(b)
    b = torch.tile(b, (o_h * o_w, 1))
    return b.T.reshape(n_out * o_h * o_w)


def circulant_like(row, n_rows, stride=1):
    """
    for stride==1, reduces to upper triangular toeplitz with n_rows and first_row==row
    Args
        row: first row
        n_rows: number of rows of resulting matrix
        stride: shift amount in subsequent rows
    """
    mat = np.zeros((n_rows, len(row)))
    for i in range(n_rows):
        mat[i, :] = np.roll(row, i*stride)
    return mat


def conv2d_circulant_like_1_ch(kernel, input_size, stride=1):
    # shapes
    k_h, k_w = kernel.shape
    i_h, i_w = input_size

    assert ((i_h - k_h) % stride == 0) and ((i_w - k_w) % stride == 0)
    o_h, o_w = (i_h-k_h)//stride+1, (i_w-k_w)//stride+1

    # construct 1d conv circulant-like matrices for each row of the kernel
    circulants = []
    for r in range(k_h):
        circulants.append(circulant_like(np.pad(kernel[r], (0, i_w - k_w)), o_w, stride=stride))

    # construct circulant-like matrix of circulant matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = circulants[0].shape

    W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block))

    for i, B in enumerate(circulants):
        for j in range(o_h):
            W_conv[j, :, i+j*stride, :] = B

    W_conv.shape = (h_blocks*h_block, w_blocks*w_block)

    return W_conv


def conv2d_toeplitz_1_ch(kernel, input_size):
    # shapes
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    o_h, o_w = i_h-k_h+1, i_w-k_w+1

    # construct 1d conv toeplitz matrices for each row of the kernel
    toeplitz = []
    for r in range(k_h):
        toeplitz.append(linalg.toeplitz(c=(kernel[r,0], *np.zeros(i_w-k_w)), r=(*kernel[r], *np.zeros(i_w-k_w))) )

    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = toeplitz[0].shape

    W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block))

    for i, B in enumerate(toeplitz):
        for j in range(o_h):
            W_conv[j, :, i+j, :] = B

    W_conv.shape = (h_blocks*h_block, w_blocks*w_block)

    return W_conv


def conv2d_circulant_like_multi_ch(kernels, input_size, stride=1, return_ls=False):
    """
    Args
      kernels: shape=(n_in, k_h, k_w)
      input_size: shape=(i_h, i_w)
      :return output mat: shape=(o_h*o_w, n_in*i_h*i_w) if not return_ls
            else output a list of n_in mats with shape=(o_h*o_w, i_h*i_w)
    """
    T_ls = []
    for k in kernels:
        T_ls.append(conv2d_circulant_like_1_ch(k, input_size, stride=stride))

    if return_ls:
        return T_ls
    else:
        return np.concatenate(T_ls, axis=1)


def test_conv2d_toeplitz(kernel, input, bias=None, padding=0):
    """Compute 2d convolution over multiple channels via toeplitz matrix
    Args:
        kernel: shape=(n_out, n_in, H_k, W_k)
        input: shape=(n_in, H_i, W_i)
        bias: shape=(n_out)
    """
    if padding > 0:
        inp = []
        for i in range(input.shape[0]):
            inp.append(np.pad(input[i, :, :], padding))
        input = np.stack(inp, axis=0)

    kernel_size = kernel.shape
    input_size = input.shape

    output_size = (kernel_size[0], input_size[1] - (kernel_size[1]-1), input_size[2] - (kernel_size[2]-1))
    output = np.zeros(output_size)

    for i,ks in enumerate(kernel):  # loop over output channel
        for j,k in enumerate(ks):  # loop over input channel
            T_k = conv2d_toeplitz_1_ch(k, input_size[1:])
            output[i] += T_k.dot(input[j].flatten()).reshape(output_size[1:])  # sum over input channels
    return output


def test_conv2d_circulant_like(kernel, inp, bias=None, stride=1, padding=0):
    """Compute 2d convolution over multiple channels via toeplitz matrix
    Args:
        kernel: shape=(n_out, n_in, k_h, h_w)
        input: shape=(n_in, i_h, i_w)
        bias: shape=(n_out)
    """
    if padding > 0:
        inp_ = []
        for i in range(inp.shape[0]):
            inp_.append(np.pad(inp[i, :, :], padding))
        inp = np.stack(inp_, axis=0)

    n_out, n_in, k_h, k_w = kernel.shape
    n_in, i_h, i_w = inp.shape

    assert ((i_h - k_h) % stride == 0) and ((i_w - k_w) % stride == 0)
    o_h, o_w = (i_h - k_h) // stride + 1, (i_w - k_w) // stride + 1
    output_size = (n_out, o_h, o_w)
    output = np.zeros(output_size)

    input_ls = [inp[j].flatten() for j in range(n_in)]
    inp = np.concatenate(input_ls, axis=0)

    print(inp.shape)

    for i,ks in enumerate(kernel):  # loop over output channel
        T = conv2d_circulant_like_multi_ch(ks, (i_h, i_w), stride=stride, return_ls=False)
        output[i, :, :] = T.dot(inp).reshape(output_size[1:])
    return output


if __name__ == "__main__":
    # ## test 1
    # ker = np.random.randn(6, 3, 3, 3)
    # inp = np.random.randn(3, 100, 100)
    # output = test_conv2d_circulant_like(ker, inp, padding=0, stride=1)
    # toutput = torch.nn.functional.conv2d(torch.tensor(inp), torch.tensor(ker), bias=None, stride=1, padding=0, dilation=1, groups=1)
    # print("test 1 error: ", np.sum((output-toutput.detach().numpy())**2))
    #
    # ## test 2 with padding==1
    # ker = np.random.randn(6, 3, 3, 3)
    # inp = np.random.randn(3, 100, 100)
    # stride, padding = 1, 1
    # output = test_conv2d_circulant_like(ker, inp, bias=None, stride=stride, padding=padding)
    # toutput = torch.nn.functional.conv2d(torch.tensor(inp), torch.tensor(ker), bias=None, stride=stride, padding=padding,
    #                                      dilation=1, groups=1)
    # print("test 2 error: ", np.sum((output - toutput.detach().numpy()) ** 2))
    #
    #
    # ## test 3 stride==2, padding==0
    # ker = np.random.randn(1, 1, 3, 3)
    # inp = np.random.randn(1, 99, 99)
    # stride = 2
    #
    # W_conv = conv2d_circulant_like_1_ch(ker[0, 0, :, :], inp.shape[1:], stride=stride)
    #
    # i_h, i_w = inp.shape[1:]
    # k_h, k_w = ker.shape[2:]
    # o_h, o_w = (i_h - k_h) // stride + 1, (i_w - k_w) // stride + 1
    #
    # m_out = W_conv.dot(inp[0].flatten()).reshape((o_h, o_w))
    # t_out =  torch.nn.functional.conv2d(torch.tensor(inp), torch.tensor(ker), bias=None, stride=stride)
    # t_out = np.squeeze(t_out)
    #
    # print("test error 3: {}".format(np.sum((m_out - t_out.detach().numpy())**2)))
    #
    #
    # ## test 4 with stride==2, padding==1
    # ker = np.random.randn(6, 3, 3, 3)
    # inp = np.random.randn(3, 99, 99)
    # stride, padding = 2, 1
    # moutput = test_conv2d_circulant_like(ker, inp, bias=None, stride=stride, padding=padding)
    # toutput = torch.nn.functional.conv2d(torch.tensor(inp), torch.tensor(ker), bias=None, stride=stride,
    #                                      padding=padding,
    #                                      dilation=1, groups=1)
    # print("test 4 error: ", np.sum((moutput - toutput.detach().numpy()) ** 2))
    #
    # ## test 5 with stride==2, padding==2
    # ker = np.random.randn(6, 3, 3, 3)
    # inp = np.random.randn(3, 99, 99)
    # stride, padding = 2, 2
    # output = test_conv2d_circulant_like(ker, inp, bias=None, stride=stride, padding=padding)
    # toutput = torch.nn.functional.conv2d(torch.tensor(inp), torch.tensor(ker), bias=None, stride=stride,
    #                                      padding=padding,
    #                                      dilation=1, groups=1)
    # print("test 5 error: ", np.sum((output - toutput.detach().numpy()) ** 2))

    ## test 6: transposed convolution (single-channel)
    ker = np.random.randn(3, 3)
    inp = np.random.randn(10, 10)
    stride, padding = 1, 0
    T_k = conv2d_toeplitz_1_ch(ker, (10, 10))
    out = np.matmul(T_k, inp.reshape(-1))
    output = np.matmul(T_k.transpose(), out)
    toutput = torch.nn.functional.conv_transpose2d(torch.tensor(out.reshape(1, 1, 8, 8)), torch.tensor(ker.reshape(1, 1, 3, 3)))
    print(np.sum((output - toutput.detach().numpy().reshape(-1)) ** 2))


    ## test 7: conv2d with bias (single-output channel)
    ker = np.random.randn(2, 3, 3)
    b = np.random.randn(1)
    inp = np.random.randn(2, 10, 10)
    stride, padding = 1, 0
    T_k = conv2d_circulant_like_multi_ch(ker, (10, 10))
    o_h, o_w = resolve_padded_dims(*inp.shape[1:], *ker.shape[1:], stride, padding)
    output = np.matmul(T_k, inp.reshape(-1)) + conv2d_bias(torch.tensor(b), o_h, o_w).numpy()
    toutput = torch.nn.functional.conv2d(torch.tensor(inp.reshape(1, 2, 10, 10)),
                                                   torch.tensor(ker.reshape(1, 2, 3, 3)),
                                         bias=torch.tensor(b))
    print(np.sum((output - toutput.detach().numpy().reshape(-1)) ** 2))

