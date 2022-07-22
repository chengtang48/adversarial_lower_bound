# pytorch implemetation
import numpy as np
import argparse
import torch
import torchvision.datasets as dts
import torchvision.transforms as trnsfrms
import torch.nn as nn

# TODO: check implementation details: apply ReLU at output layer? apply bias at output layer?

class FCModel(nn.Module):
    def __init__(self, network_params=(10, 10, 10), d_in=28*28):
        super(FCModel,self).__init__()
        self.linears = []
        self.d_in = d_in
        d_in_curr = d_in
        for i, d_out in enumerate(network_params):
            if i == (len(network_params) - 1):
                lin_layer = nn.Linear(d_in_curr, d_out, bias=False)
                #lin_layer = nn.Linear(d_in_curr, d_out, bias=True)
            else:
                lin_layer = nn.Linear(d_in_curr, d_out)
            self.linears.append(lin_layer)
            self.add_module("linear_{}".format(i), lin_layer)
            d_in_curr = d_out
        self.relu = nn.ReLU()


    def forward(self, image):
        a = image.view(-1, self.d_in)
        for idx, linear in enumerate(self.linears):
            if idx < len(self.linears)-1:
                a = self.relu(linear(a))
            else:
                a = linear(a) # last layer
        return a


def train(fcmodel, train_dataloader, criter='CE', l_r=0.001, numepchs=5):
    loss = None
    if criter == 'CE':
        loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(fcmodel.parameters(), lr=l_r)
    nttlstps = len(train_dataloader)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
    torch.device('cpu')
    for epoch in range(numepchs):
        for i, (imgs, lbls) in enumerate(train_dataloader):
            #lr_scheduler.step()
            imgs = imgs.reshape(-1, fcmodel.d_in)

            outp = fcmodel(imgs)
            losses = loss(outp, lbls)

            optim.zero_grad()
            losses.backward()
            optim.step()

            if (i + 1) % 100 == 0:
                print('Epochs [{}/{}], Step[{}/{}], Losses: {}'.format(epoch + 1, numepchs, i + 1,
                                                                       nttlstps, losses.item()))


def test(fcmodel, test_dataloader):
    c_err = 0
    n_data = 0
    for imgs, lbls in test_dataloader:
        outp = fcmodel(imgs)
        preds = np.argmax(outp.detach(), axis=1)
        for p, l in zip(preds, lbls):
            c_err += int(p!=l)
            n_data += 1

    print("Test acc: ", 1-c_err / n_data)


if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', default=10)
    parser.add_argument('--model_config')
    parser.add_argument('--save_model', action='store_true',
                        help='whether to save model to path')
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

    print(fcmodel)
    print(fcmodel.parameters())

    trnsform = trnsfrms.Compose([trnsfrms.ToTensor(), trnsfrms.Normalize((0.7,), (0.7,))])

    mnist_trainset = dts.MNIST(root='./data', train=True, download=True, transform=trnsform)
    trainldr = torch.utils.data.DataLoader(mnist_trainset, batch_size=int(args.train_batch_size), shuffle=True)

    mnist_testset = dts.MNIST(root='./data', train=False, download=True, transform=trnsform)
    testldr = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

    train(fcmodel, trainldr, criter='CE', l_r=0.0001, numepchs=10)

    test(fcmodel, testldr)

    if args.save_model:
        print("Saving trained model to {}".format(args.model_path))
        torch.save(fcmodel.state_dict(), args.model_path)




