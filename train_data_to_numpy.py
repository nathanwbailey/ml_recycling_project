import torch
import torchvision
import numpy as np
from PIL import Image
from torch.nn import functional
from itertools import chain
from sklearn.utils import shuffle
import tensorflow.lite as tflite
from dataset import RecyclingDataset

mean = torch.Tensor([0.6661, 0.6211, 0.5492]).to('cpu')
std = torch.Tensor([0.2871, 0.2917, 0.3310]).to('cpu')

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

train_dataset = RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)


train_image = []
for idx, batch in enumerate(trainloader):
    train_image.append(batch[0].permute(0,2,3,1))

train_image = torch.cat(train_image)
np.savez('train_data.npz', train_image)