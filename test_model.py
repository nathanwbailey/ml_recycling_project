import torch
import torchvision
import sys
import dataset

import network
from PIL import Image


def calculate_accuracy(outputs, ground_truth):
    softmaxed_output = torch.nn.functional.softmax(outputs, dim=1)
    predictions = torch.argmax(softmaxed_output, dim=1)
    num_correct = torch.sum(torch.eq(predictions, ground_truth)).item()
    return num_correct, ground_truth.size(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mean = torch.Tensor([0.6661, 0.6211, 0.5492])
std = torch.Tensor([0.2871, 0.2917, 0.3310])

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])


model = network.RecycleNetwork(num_classes=5).to(device)
model.load_state_dict(torch.load('recycle_net_trained.pt'))
model.eval()

dataset_to_use = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='test', data_transforms=transforms)

testloader = torch.utils.data.DataLoader(dataset_to_use, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

num_examples = 0
num_correct = 0
with torch.no_grad():
    for batch in testloader:
        images = batch[0].to(device)
        labels = batch[1].to(device)
        output = model(images)
        softmaxed_output = torch.nn.functional.softmax(output, dim=1)
        predictions = torch.argmax(softmaxed_output, dim=1)
        num_corr, num_ex = calculate_accuracy(output, labels)
        num_examples += num_ex
        num_correct += num_corr

print('Test Accuracy: {:.4f}'.format(num_correct/num_examples))