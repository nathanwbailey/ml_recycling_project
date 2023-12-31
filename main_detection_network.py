import torch
import torchvision
import sys
from dataset_detection_network import DetectionDataset
from detection_network import DetectionNetwork
from train_test_detection_network import train_network, test_network

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor()])

train_dataset = DetectionDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, pin_memory=True)


mean = torch.zeros(3).to(device)
std = torch.zeros(3).to(device)

for idx, batch in enumerate(trainloader):
    image = batch[0].to(device)
    image_mean = torch.mean(image, dim=(0,2,3))
    image_std = torch.std(image, dim=(0,2,3))
    mean = torch.add(mean, image_mean)
    std = torch.add(std, image_std)

mean = (mean/len(trainloader)).to('cpu')
std = (std/len(trainloader)).to('cpu')

print(mean)
print(std)

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

train_dataset = DetectionDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)

valid_dataset = DetectionDataset(data_dir='compiled_dataset', dataset_type='valid', data_transforms=transforms)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

test_dataset = DetectionDataset(data_dir='compiled_dataset', dataset_type='test', data_transforms=transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

model = DetectionNetwork().to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=0.01, momentum=0.9, weight_decay=5e-2)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, mode='min', patience=3, min_lr=0.0000000000001, threshold_mode='abs', threshold=1e-2, verbose=True)

#Train and testing
num_epochs = 100
model = train_network(model=model, num_epochs=num_epochs, optimizer=optimizer, loss_function=loss, trainloader=trainloader, validloader=validloader, device=device, scheduler=scheduler)
test_loss = torch.nn.CrossEntropyLoss()
test_network(model=model, testloader=testloader, loss_function=test_loss, device=device)