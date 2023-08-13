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

# for idx, batch in enumerate(trainloader):
#     image = batch[0].to(device)
#     image_mean = torch.mean(image, dim=(0,2,3))
#     image_std = torch.std(image, dim=(0,2,3))
#     mean = torch.add(mean, image_mean)
#     std = torch.add(std, image_std)

# mean = (mean/len(trainloader)).to('cpu')
# std = (std/len(trainloader)).to('cpu')

# print(mean)
# print(std)

mean = torch.Tensor([0.6662, 0.6211, 0.5492]).to('cpu')
std = torch.Tensor([0.2864, 0.2910, 0.3304]).to('cpu')

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

train_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)

valid_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='valid', data_transforms=transforms)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True)

test_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='test', data_transforms=transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

model = network.RecycleNetwork(num_classes=5).to(device)
#Loss function, optimizer, scheduler
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=0.001, momentum=0.9, weight_decay=5e-2)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, mode='min', patience=3., min_lr=0.00000000001, threshold_mode='abs', threshold=1e-2, verbose=True)

#Train and testing
num_epochs = 100
model = train_test.train_network(model=model, num_epochs=num_epochs, optimizer=optimizer, loss_function=loss, trainloader=trainloader, validloader=validloader, device=device, scheduler=scheduler)
test_loss = torch.nn.CrossEntropyLoss()
train_test.test_network(model=model, testloader=testloader, loss_function=test_loss, device=device)