import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

from helper import getUCF101
from helper import loadSequence
import resnet_3d

import h5py
import cv2

from multiprocessing import Pool

IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 24
lr = 0.0001
num_of_epochs = 10

data_directory = '/projects/training/bauh/AR/'
class_list, train, test = getUCF101(base_directory = data_directory)

### Testing

model = torch.load('3d_resnet.model')
model.avgpool = nn.AdaptiveAvgPool3d(1)

model.cuda()


acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))
mean = np.asarray([0.485, 0.456, 0.406],np.float32)
std = np.asarray([0.229, 0.224, 0.225],np.float32)
model.eval()


preds = []
for i in range(len(test[0])):

    t1 = time.time()

    index = random_indices[i]

    filename = test[0][index]
    filename = filename.replace('.avi','.hdf5')
    filename = filename.replace('UCF-101','UCF-101-hdf5')

    h = h5py.File(filename,'r')
    nFrames = len(h['video'])

    data = np.zeros((1, nFrames,3,IMAGE_SIZE,IMAGE_SIZE),dtype=np.float32)

    for j in range(nFrames):
        frame = h['video'][j]
        frame = frame.astype(np.float32)
        frame = cv2.resize(frame,(IMAGE_SIZE,IMAGE_SIZE))
        frame = frame/255.0
        frame = (frame - mean)/std
        frame = frame.transpose(2,0,1)
        data[0, j,:,:,:] = frame
    h.close()

    # prediction = np.zeros((1, nFrames,NUM_CLASSES),dtype=np.float32)

    # loop_i = list(range(0,nFrames,200))
    # loop_i.append(nFrames)

    # for j in range(len(loop_i)-1):
    #     data_batch = data[loop_i[j]:loop_i[j+1]]

    #     with torch.no_grad():
    #         x = np.asarray(data_batch,dtype=np.float32)
    #         x = Variable(torch.FloatTensor(x)).cuda().contiguous()

    #         output = model(x)

    #     prediction[loop_i[j]:loop_i[j+1]] = output.cpu().numpy()
    with torch.no_grad():
        x = np.asarray(data,dtype=np.float32)
        x = np.transpose(x, [0, 2, 1, 3, 4])
        x = Variable(torch.FloatTensor(x)).cuda().contiguous()
        output = model(x)
        del x
    prediction = output.cpu().numpy()
    del output
    prediction = prediction[0]

    # softmax
    for j in range(prediction.shape[0]):
        prediction[j] = np.exp(prediction[j])/np.sum(np.exp(prediction[j]))

    prediction = np.mean(np.log(prediction),axis=0)
    preds.append(prediction)
    print("idx:", i, "time", time.time()-t1)

preds = np.array(preds)
np.save("seq_preds.npy", preds)