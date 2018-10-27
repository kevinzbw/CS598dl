import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms



import os
import argparse

from gan import Generator, Discriminator
from utils import plot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training GAN')
parser.add_argument('--with_g', action='store_true')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

batch_size = 128

print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

print('==> Loading the network')
if args.with_g:
    model = torch.load('model/discriminator.model')
else:
    model = torch.load('model/cifar10.model')
model = model.to(device)
model.eval()

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

batch_idx, (X_batch, Y_batch) = next(testloader)

X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).to(device)

layers = [4, 8]
for layer in layers:
    lr = 0.1
    weight_decay = 0.001
    for i in range(200):
        _, output = model(X, layer)

        loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                grad_outputs=torch.ones(loss.size()).to(device),
                                create_graph=True, retain_graph=False, only_inputs=True)[0]

        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
        print(i,accuracy,-loss)

        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0

    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples[0:100])
    filename = 'visualization/max_features_wt_g_{}.png'.format(layer) if args.with_g else 'visualization/max_features_wo_g_{}.png'.format(layer)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)