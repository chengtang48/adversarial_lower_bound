import numpy as np
import torch
import torch.nn as nn
import argparse
import torchvision.datasets as dts
import torchvision.transforms as trnsfrms

from main import create_init_bounds
from fast_appx import fast_layer_matrix_form, fast_conv2D_layer_matrix_form, fast_pool2D_layer_matrix_form, find_last_layer_bound


class LeNet5(nn.Module):
    """
    LeNet-5 implementation
    ref source: http://d2l.ai/chapter_convolutional-neural-networks/lenet.html
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10))


    def forward(self, input):
        # input shape = batch_size by in_channels by h_k by w_k
        return self.net(input)


def train(lenet_model, train_dataloader, criter='CE', l_r=0.001, numepchs=5):
    loss = None
    if criter == 'CE':
        loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(lenet_model.parameters(), lr=l_r)
    nttlstps = len(train_dataloader)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
    torch.device('cpu')
    for epoch in range(numepchs):
        for i, (imgs, lbls) in enumerate(train_dataloader):
            #lr_scheduler.step()
            #imgs = imgs.reshape(-1, fcmodel.d_in)
            outp = lenet_model(imgs)
            losses = loss(outp, lbls)

            optim.zero_grad()
            losses.backward()
            optim.step()

            if (i + 1) % 100 == 0:
                print('Epochs [{}/{}], Step[{}/{}], Losses: {}'.format(epoch + 1, numepchs, i + 1,
                                                                       nttlstps, losses.item()))


def test(lenet_model, test_dataloader):
    c_err = 0
    n_data = 0
    for imgs, lbls in test_dataloader:
        outp = lenet_model(imgs)
        preds = np.argmax(outp.detach(), axis=1)
        for p, l in zip(preds, lbls):
            c_err += int(p!=l)
            n_data += 1

    print("Test acc: ", 1-c_err / n_data)


def get_lenet_hidden_layer_lower_bound(model,  x,  eps):
    L,  U = create_init_bounds(x, eps)
    L, U = torch.tensor(np.array(L).reshape(1, 1, x.shape[1], x.shape[2])), \
           torch.tensor(np.array(U).reshape(1, 1, x.shape[1], x.shape[2]))
    param_lst = list(model.parameters())
    inc_j = -1
    for i in range(7):
        #print(i, L.shape, U.shape)
        if i == 0:
            W_k, b_k = param_lst[i], param_lst[i + 1]
            stride, padding = 1, 2
            L, U = fast_conv2D_layer_matrix_form((W_k, b_k), (stride, padding), L, U, activation=nn.Sigmoid())
        elif i == 2:
            W_k, b_k = param_lst[i], param_lst[i + 1]
            stride, padding = 1, 0
            L, U = fast_conv2D_layer_matrix_form((W_k, b_k), (stride, padding), L, U, activation=nn.Sigmoid())
        if i == 1 or i == 3:
            # avg pooling
            L, U = fast_pool2D_layer_matrix_form(2, (2, 0), L, U, op='avg')
        elif i == 4:
            # flattening
            L, U = L.reshape(-1), U.reshape(-1)
        elif i >= 5:
            # linear layers 5 -->  4, 5 | 6 --> 6, 7 #|  7 --> 8, 9
            W, b = param_lst[i + inc_j], param_lst[i + inc_j + 1]
            L, U = fast_layer_matrix_form((W, b), L, U, activation=nn.Sigmoid())
            inc_j += 1
    return L, U


if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model', action='store_true',
                        help='whether to save model to path')
    parser.add_argument('--train', action='store_true',
                        help='train model')
    parser.add_argument('--model_path')

    args = parser.parse_args()

    model = LeNet5()

    print(model)
    print(model.parameters())

    trnsform = trnsfrms.Compose([trnsfrms.ToTensor(), trnsfrms.Normalize((0.7,), (0.7,))])

    mnist_trainset = dts.MNIST(root='./data', train=True, download=True, transform=trnsform)
    trainldr = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

    mnist_testset = dts.MNIST(root='./data', train=False, download=True, transform=trnsform)
    testldr = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

    if args.train:
        train(model, trainldr, criter='CE', l_r=0.001, numepchs=5)
        test(model, testldr)

    if args.save_model:
        print("Saving trained model to {}".format(args.model_path))
        torch.save(model.state_dict(), args.model_path)

    ### test adv lower bounds
    eps = 0.008
    model = LeNet5()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    W_o, b_o = list(model.parameters())[-2], list(model.parameters())[-1]
    total_success = 0
    n_samples = 0
    for images, labels in testldr:
        preds = model(images).detach()
        preds_id = np.argmax(preds, axis=1)
        for idx, (pred_id, label) in enumerate(zip(preds_id, labels)):
            #if pred_id == label:
            outp = preds[idx]
            max_idx = np.argmax(outp)
            x_0 = images[idx].detach().numpy()
            L, U = get_lenet_hidden_layer_lower_bound(model, x_0, eps)
            #print("original pred: ", outp)
            #print("max idx: ", max_idx)
            L_f = find_last_layer_bound(W_o, b_o, max_idx, L, U)
            if np.min(L_f.detach().numpy()) < 0:
                total_success += 1
                #print("success")
            else:
                print("failed")
            n_samples += 1

    print("successes/samples: {} / {} ".format(total_success, n_samples))




    #get_lenet_lower_bound(model, None, None)
