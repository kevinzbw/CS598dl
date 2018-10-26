import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms


import time
import os
import argparse
import numpy as np

from gan import Generator, Discriminator
from utils import plot

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training GAN')
# parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

aD =  Discriminator()
aG = Generator()

aD = aD.to(device)
aG = aG.to(device)

if device == 'cuda':
    aD = torch.nn.DataParallel(aD)
    aG = torch.nn.DataParallel(aG)
    cudnn.benchmark = True

learning_rate = 0.0001

criterion = nn.CrossEntropyLoss()
optimizer_g = torch.optim.Adam(aG.parameters(), lr=learning_rate, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=learning_rate, betas=(0,0.9))

def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

n_classes = 10
n_z = 100
gen_train = 5

np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0,1,(100,n_z))
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).to(device)

def check_grad(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if('step' in state and state['step']>=1024):
                state['step'] = 1000

def train(epoch):
    print('\nTraining Epoch: %d' % epoch)
    # Train the model
    loss1, loss2, loss3, loss4, loss5 = [], [], [], [], []
    acc1 = []
    aG.train()
    aD.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        if(Y_train_batch.shape[0] < batch_size):
            continue
        # train G
        if((batch_idx%gen_train)==0):
            for p in aD.parameters():
                p.requires_grad_(False)
            for p in aG.parameters():
                p.requires_grad_(True)

            aG.zero_grad()

            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).to(device)
            fake_label = Variable(torch.from_numpy(label)).to(device)

            fake_data = aG(noise)
            gen_source, gen_class  = aD(fake_data)

            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label)

            gen_cost = -gen_source + gen_class
            gen_cost.backward()

            check_grad(optimizer_g)
            optimizer_g.step()

        # train D
        if((batch_idx%gen_train)==0):
            for p in aD.parameters():
                p.requires_grad_(True)
            for p in aG.parameters():
                p.requires_grad_(False)

        aD.zero_grad()

        # train discriminator with input from generator
        label = np.random.randint(0,n_classes,batch_size)
        noise = np.random.normal(0,1,(batch_size,n_z))
        label_onehot = np.zeros((batch_size,n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).to(device)
        fake_label = Variable(torch.from_numpy(label)).to(device)
        with torch.no_grad():
            fake_data = aG(noise)

        disc_fake_source, disc_fake_class = aD(fake_data)

        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)

        # train discriminator with input from the discriminator
        real_data = Variable(X_train_batch).to(device)
        real_label = Variable(Y_train_batch).to(device)

        disc_real_source, disc_real_class = aD(real_data)

        prediction = disc_real_class.data.max(1)[1]
        accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0

        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)

        gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)

        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        disc_cost.backward()

        check_grad(optimizer_d)
        optimizer_d.step()

        # loss
        loss1.append(gradient_penalty.item())
        loss2.append(disc_fake_source.item())
        loss3.append(disc_real_source.item())
        loss4.append(disc_real_class.item())
        loss5.append(disc_fake_class.item())
        acc1.append(accuracy)
        if((batch_idx%50)==0):
            print(epoch, batch_idx, "%.2f" % np.mean(loss1), 
                                    "%.2f" % np.mean(loss2), 
                                    "%.2f" % np.mean(loss3), 
                                    "%.2f" % np.mean(loss4), 
                                    "%.2f" % np.mean(loss5), 
                                    "%.2f" % np.mean(acc1))
        
def test(epoch):
    print('\nTesting Epoch: %d' % epoch)
    # Test the model
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()

            with torch.no_grad():
                _, output = aD(X_test_batch)

            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    print('Testing', accuracy_test)

def eval_noise():
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)
        aG.train()

    fig = plot(samples)
    plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
    plt.close(fig)
    
for epoch in range(start_epoch, start_epoch+200):
    start_time = time.time()
    train(epoch)
    test(epoch)
    print("Time", time.time()-start_time)
    eval_noise()
    if(((epoch+1)%1)==0):
        torch.save(aG,'model/tempG.model')
        torch.save(aD,'model/tempD.model')

torch.save(aG,'model/generator.model')
torch.save(aD,'model/discriminator.model')