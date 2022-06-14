# pytorch implemetation
import numpy as np
import argparse
import torch
import torchvision.datasets as dts
import torchvision.transforms as trnsfrms
import torch.nn as nn


class classificationmodel(nn.Module):
    def __init__(self, network_params=(10, 10, 10), d_in=28*28):
        super(classificationmodel,self).__init__()
        self.linears = []
        self.d_in = d_in
        for i, d_out in enumerate(network_params):
            if i == (len(network_params) - 1):
                lin_layer = nn.Linear(d_in, d_out, bias=False)
            else:
                lin_layer = nn.Linear(d_in, d_out)
            self.linears.append(lin_layer)
            self.add_module("linear_{}".format(i), lin_layer)
            d_in = d_out
        self.relu = nn.ReLU()

    def forward(self, image):
        a = image.view(-1, self.d_in)
        for linear in self.linears:
            a = self.relu(linear(a))
        return a


def train(cmodel, train_dataloader, d_in=28*28, criter='CE', l_r=0.001, numepchs=5):
    loss = None
    if criter == 'CE':
        loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(cmodel.parameters(), lr=l_r)
    nttlstps = len(train_dataloader)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    torch.device('cpu')
    for epoch in range(numepchs):
        for i, (imgs, lbls) in enumerate(train_dataloader):
            #lr_scheduler.step()
            imgs = imgs.reshape(-1, d_in)

            outp = cmodel(imgs)
            losses = loss(outp, lbls)

            optim.zero_grad()
            losses.backward()
            optim.step()

            if (i + 1) % 100 == 0:
                print('Epochs [{}/{}], Step[{}/{}], Losses: {}'.format(epoch + 1, numepchs, i + 1,
                                                                       nttlstps, losses.item()))


def test(cmodel, test_dataloader):
    c_err = 0
    n_data = 0
    for imgs, lbls in test_dataloader:
        outp = cmodel(imgs)
        preds = np.argmax(outp.detach(), axis=1)
        for p, l in zip(preds, lbls):
            c_err += int(p!=l)
            n_data += 1

    print("Test acc: ", 1-c_err / n_data)


if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model', action='store_true',
                        help='whether to save model to path')
    parser.add_argument('--model_path')
    args = parser.parse_args()

    cmodel = classificationmodel()

    print(cmodel)
    print(cmodel.parameters())

    trnsform = trnsfrms.Compose([trnsfrms.ToTensor(), trnsfrms.Normalize((0.7,), (0.7,))])

    mnisttrainset = dts.MNIST(root='./data', train=True, download=True, transform=trnsform)
    trainldr = torch.utils.data.DataLoader(mnisttrainset, batch_size=10, shuffle=True)

    mnist_testset = dts.MNIST(root='./data', train=False, download=True, transform=trnsform)
    testldr = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

    train(cmodel, trainldr, d_in=28*28, criter='CE', l_r=0.0005, numepchs=10)

    test(cmodel, testldr)

    if args.save_model:
        print("Saving trained model to {}".format(args.model_path))
        torch.save(cmodel.state_dict(), args.model_path)




