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

test_dataset = RecyclingDataset(data_dir='compiled_dataset', dataset_type='test', data_transforms=transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, pin_memory=True)


test_image = []
test_label = []
for idx, batch in enumerate(testloader):
    test_image.append(batch[0].permute(0,2,3,1))
    test_label.append(batch[1])

test_image = torch.cat(test_image)
test_label = torch.cat(test_label)
np.savez('test_data.npz', test_image)
np.savez('test_label.npz', test_label)