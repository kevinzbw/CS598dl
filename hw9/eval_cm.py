from helper import getUCF101
import numpy as np

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