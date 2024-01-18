import torch
import torchvision
import sys
import dataset
from network import RecycleNetwork
import train_test


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor()])

train_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, pin_memory=True)


mean = torch.Tensor([0.6661, 0.6211, 0.5492]).to('cpu')
std = torch.Tensor([0.2871, 0.2917, 0.3310]).to('cpu')

# mean = torch.zeros(3).to(device)
# std = torch.zeros(3).to(device)

# for idx, batch in enumerate(trainloader):
#     image = batch[0].to(device)
#     image_mean = torch.mean(image, dim=(0,2,3))
#     image_std = torch.std(image, dim=(0,2,3))
#     mean = torch.add(mean, image_mean)
#     std = torch.add(std, image_std)

# mean = (mean/len(trainloader)).to('cpu')
# std = (std/len(trainloader)).to('cpu')

print(mean)
print(std)

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

transforms_augment_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)),     torchvision.transforms.RandomHorizontalFlip(p=1.0), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

transforms_augment_2 = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)),     torchvision.transforms.RandomVerticalFlip(p=1.0), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])


transforms_augment_3 = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)),     torchvision.transforms.RandomRotation(degrees=50), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

transforms_augment_4 = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)),     torchvision.transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

transforms_augment_5 = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)),     torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

train_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms)

train_dataset_augmented_1 = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms_augment_1)

train_dataset_augmented_2 = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms_augment_2)

train_dataset_augmented_3 = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms_augment_3)

train_dataset_augmented_4 = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms_augment_4)

train_dataset_augmented_5 = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms_augment_5)

train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_augmented_1, train_dataset_augmented_2, train_dataset_augmented_3, train_dataset_augmented_4, train_dataset_augmented_5])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)

print(len(trainloader))

valid_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='valid', data_transforms=transforms)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

test_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='test', data_transforms=transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

model = RecycleNetwork(num_classes=5).to(device)
#Loss function, optimizer, scheduler
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=0.0001, momentum=0.9, weight_decay=5e-2)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, mode='min', patience=3, min_lr=0.0000000000001, threshold_mode='abs', threshold=1e-2, verbose=True)

#Train and testing
num_epochs = 100
model = train_test.train_network(model=model, num_epochs=num_epochs, optimizer=optimizer, loss_function=loss, trainloader=trainloader, validloader=validloader, device=device, scheduler=scheduler)
test_loss = torch.nn.CrossEntropyLoss()
train_test.test_network(model=model, testloader=testloader, loss_function=test_loss, device=device)