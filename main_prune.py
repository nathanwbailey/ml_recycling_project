import torch
import torchvision
import sys
import dataset
from network import RecycleNetwork
import train_test
import torch_pruning as pruning
import math
import pytorch_model_summary as pms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

mean = torch.Tensor([0.6661, 0.6211, 0.5492]).to('cpu')
std = torch.Tensor([0.2871, 0.2917, 0.3310]).to('cpu')

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

transforms_augment_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)),     torchvision.transforms.RandomHorizontalFlip(p=1.0), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

transforms_augment_2 = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)),     torchvision.transforms.RandomVerticalFlip(p=1.0), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

transforms_augment_3 = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)),     torchvision.transforms.RandomRotation(degrees=50), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

transforms_augment_4 = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)),     torchvision.transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

transforms_augment_5 = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)),     torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])


def train_network(model, device, epochs):
    train_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms)
    
    train_dataset_augmented_1 = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms_augment_1)

    train_dataset_augmented_2 = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms_augment_2)

    train_dataset_augmented_3 = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms_augment_3)

    train_dataset_augmented_4 = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms_augment_4)

    train_dataset_augmented_5 = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms_augment_5)

    train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_augmented_1, train_dataset_augmented_2, train_dataset_augmented_3, train_dataset_augmented_4, train_dataset_augmented_5])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
    
    valid_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='valid', data_transforms=transforms)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
    
    model.to(device)
    model.train()
    for param in model.parameters():
        param.requires_grad=True
    
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=0.00001, momentum=0.9, weight_decay=5e-2)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, mode='min', patience=3, min_lr=0.00000000001, threshold_mode='abs', threshold=1e-2, verbose=True)

    model = train_test.train_network(model=model, num_epochs=epochs, optimizer=optimizer, loss_function=loss, trainloader=trainloader, validloader=validloader, device=device, scheduler=scheduler)
    return model



model = RecycleNetwork(num_classes=5)

model.load_state_dict(torch.load('recycle_net_trained.pt'))
model.train()


num_to_prune=0.2
DG = pruning.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,150,150))
model.to(device)
num_conv_layers_pruned = 0
train_model=True
for idx, (name,layer) in enumerate(model.named_modules()):
    if isinstance(layer, torch.nn.Conv2d):
        num_conv_layers_pruned +=1
        train_model=True
        prune_idx = torch.argsort(torch.sum(torch.abs(layer.weight.data), dim=(1,2,3))).tolist()
        num = math.ceil(len(prune_idx)*num_to_prune)
        prune_idx = prune_idx[:num]
        if layer.groups == layer.in_channels:
            print('Depthwise Detected')
            group = DG.get_pruning_group(layer, pruning_fn=pruning.prune_depthwise_conv_out_channels, idxs=prune_idx)
            if DG.check_pruning_group(group):
                group.prune()
        else:
            group = DG.get_pruning_group(layer, pruning_fn=pruning.prune_conv_out_channels, idxs=prune_idx)
            if DG.check_pruning_group(group):
                group.prune()
    if num_conv_layers_pruned % 3 == 0 and num_conv_layers_pruned != 0 and train_model:
        if num_conv_layers_pruned == 3:
            train_model=False
            continue
        print('Starting Training Cycle')
        pms.summary(model, torch.zeros((1, 3, 150,150)).to(device), show_input=False, print_summary=True, max_depth=5, show_parent_layers=True)
        train_network(model, device=device, epochs=100)
        print('Ended Training Cycle')
        train_model=False
        




test_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='test', data_transforms=transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

#Train and testing
model.train()
model.to(device)
num_epochs = 100
model = train_network(model, device=device, epochs=num_epochs)
test_loss = torch.nn.CrossEntropyLoss()
model.eval()
train_test.test_network(model=model, testloader=testloader, loss_function=test_loss, device=device)