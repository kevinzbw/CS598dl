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
from helper import loadFrame

import h5py
import cv2

from multiprocessing import Pool
import numpy as np

IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 100
lr = 0.0001
num_of_epochs = 10


data_directory = '/projects/training/bauh/AR/'
class_list, train, test = getUCF101(base_directory = data_directory)

confusion = np.load('single_frame_confusion_matrix.npy')
labels = np.array(class_list)

for i in range(confusion.shape[0]):
    idx = np.argsort(confusion[i])
    m = confusion[i, idx[-2]] if idx[-1] == i else confusion[i, idx[-1]]
    confusion[i] -= m

diag = np.diag(confusion)
result = sorted(zip(diag, labels))
print(result)
for val, lab in result:
    print(lab)
