import torch
from training import dataset
import numpy as np

img_dataset = dataset.ImageFolderDataset(".datasets/cifar_train.zip", use_labels=False)
mean = 0
std = 0
for img, _ in img_dataset:
    x = img/127.5 - 1
    mean += x
    std += np.std(x)
mean /= len(img_dataset)
std /= len(img_dataset)
np.save(".datasets/cifar_mean", mean)
np.save(".datasets/cifar_std", std)
print(mean, std)