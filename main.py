import torch
import torchvision
import sys
import dataset
import network
import train_test


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)), torchvision.transforms.ToTensor()])

train_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, pin_memory=True)
mean = torch.zeros(3).to(device)
std = torch.zeros(3).to(device)

for idx, batch in enumerate(trainloader):
    image = batch[0].to(device)
    image_mean = torch.mean(image, dim=(0,2,3))
    image_std = torch.std(image, dim=(0,2,3))
    mean = torch.add(mean, image_mean)
    std = torch.std(std, image_std)

mean = (mean/len(trainloader)).to('cpu')
std = (std/len(trainloader)).to('cpu')

transforms = torch.transforms.Compose([torchvision.transforms.Resize((224,224)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

train_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)

valid_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='valid', data_transforms=transforms)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True)

test_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='test', data_transforms=transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

model = network.RecycleNetwork().to(device)
#Loss function, optimizer, scheduler


#Train and testing