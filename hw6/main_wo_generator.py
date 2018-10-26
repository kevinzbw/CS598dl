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


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training GAN')
# parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

batch_size = 128

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

print('==> Loading the network')
net =  Discriminator()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


test_acc_ep = []
test_loss_ep = []

def check_grad(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if('step' in state and state['step']>=1024):
                state['step'] = 1000

def train(epoch):
    print('\nTraining Epoch: %d' % epoch)
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        _, out_fc10 = net(inputs)

        loss = criterion(out_fc10, targets)
        
        optimizer.zero_grad()
        loss.backward()
        check_grad(optimizer)
        optimizer.step()

def test(epoch):
    global best_acc
    print('\nTesting Epoch: %d' % epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, out_fc10 = net(inputs)
            loss = criterion(out_fc10, targets)

            test_loss += loss.item()
            _, predicted = out_fc10.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_acc_ep.append(100.*correct/total)
        test_loss_ep.append(test_loss/(batch_idx+1))
        
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc

for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    if (epoch+1) % 20 == 0:
        torch.save(net,'model/cifar10_{}.model'.format(epoch))

test(100)
print(test_acc_ep)
print(test_loss_ep)
print(best_acc)
torch.save(net,'model/cifar10_100.model')
