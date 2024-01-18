import torch
import torchvision
import sys
import dataset
import dataset_taylor_data
from network import RecycleNetwork
import train_test
import torch_pruning as pruning
import math
import pytorch_model_summary as pms
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

mean = torch.Tensor([0.6661, 0.6211, 0.5492]).to('cpu')
std = torch.Tensor([0.2871, 0.2917, 0.3310]).to('cpu')

model = RecycleNetwork(num_classes=5).to(device)
model.load_state_dict(torch.load('recycle_net_trained.pt'))
model.eval()

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

valid_dataset = dataset_taylor_data.RecyclingDataset(data_dir='compiled_dataset', dataset_type='valid', data_transforms=transforms)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
num_batches = len(validloader)
print(num_batches)
activations = []
activation_idx = 0
grad_index = 0
filter_ranks = []
names = []
grads = []

def compute_rank(grad):
    #Gradient is graident of the loss w.r.t the output (x)
    global grad_index
    act_idx = len(activations) - grad_index -1
    rank = torch.abs(torch.mean((grad*activations[act_idx]), dim=(2,3)))
    div = torch.sqrt(torch.sum(torch.pow(rank, 2), dim=1))
    for i in range(div.size()[0]):
        rank[i] / div[i]
    rank = torch.mean(rank, dim=0)
    if len(filter_ranks) < grad_index+1:
        filter_ranks.append(rank)
    else:
        filter_ranks[grad_index] += rank
    grad_index += 1

device = 'cpu'
model.to(device)
test_loss = torch.nn.CrossEntropyLoss()
for idx, batch in enumerate(validloader):
    grad_index = 0
    activations = []
    x = batch[0].to(device)
    labels = batch[1].to(device)
    for layer, (name, module) in enumerate(model.named_modules()):
        if re.search('backbone.blocks.\d.\d.+', name) or re.search('backbone.head.\d', name) or re.search('backbone.stem.\d', name):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(compute_rank)
                activations.append(x)
                # activation_idx +=1
                if name not in names:
                    names.append(name)
        elif re.search('backbone.avgpool+', name) or re.search('backbone.fc+', name) or re.search('backbone.dropout+', name):
            x = module(x)

            if re.search('backbone.avgpool+', name):
                x = torch.flatten(x, 1)
                x.retain_grad()
    outputs = x
    loss = test_loss(outputs, labels)
    loss.backward()
    model.zero_grad()
    # x.grad.zero_()
    # del x

filter_ranks = [filter_rank/num_batches for filter_rank in filter_ranks]
names.reverse()
# print(names)
# print(len(names))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_network(model, device, epochs):
    train_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='train', data_transforms=transforms)
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
    model = train_test.train_network(model=model, num_epochs=epochs, optimizer=optimizer, loss_function=loss, trainloader=trainloader, validloader=validloader, device=device, scheduler=scheduler, patience=10)
    return model

model = RecycleNetwork(num_classes=5).to(device)
model.load_state_dict(torch.load('recycle_net_trained.pt'))
model.train()

#11 prunes
num_to_prune=0.10
DG = pruning.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,150,150).to(device))
model.to(device)
num_conv_layers_pruned = 0
train_model=True
num_prunes = 0
idx_conv = 0
print(model)
filter_ranks_to_use = filter_ranks.copy()
for idx, (name,layer) in enumerate(model.named_modules()):
    if isinstance(layer, torch.nn.Conv2d):
        num_conv_layers_pruned +=1
        train_model=True
        prune_idx = filter_ranks_to_use[len(filter_ranks_to_use)-idx_conv-1]
        # print(prune_idx.size())
        name = names[len(names)-idx_conv-1]
        # print(len(names))
        print(name)
        prune_idx = torch.argsort(prune_idx).tolist()
        # prune_idx = torch.flip(prune_idx, dims=(0,)).tolist()
        num = math.ceil(len(prune_idx)*num_to_prune)
        prune_idx = prune_idx[:num]
        print(prune_idx)
        # if len(names) > 2 and 'depthwise' in names[-1]:
        #     mask = torch.tensor([(1 if i not in prune_idx else 0) for i in range(filter_ranks_to_use[-1].size()[0])]).type('torch.BoolTensor')
        #     # print(mask)
        #     print(filter_ranks_to_use[-1].size())
        #     filter_ranks_to_use[-1] = torch.masked_select(filter_ranks_to_use[-1], mask)
        #     print(filter_ranks_to_use[-1].size())
        if layer.groups == layer.in_channels:
            print('Depthwise Detected')
            group = DG.get_pruning_group(layer, pruning_fn=pruning.prune_depthwise_conv_out_channels, idxs=prune_idx)
            if DG.check_pruning_group(group):
                group.prune()
        else:
            group = DG.get_pruning_group(layer, pruning_fn=pruning.prune_conv_out_channels, idxs=prune_idx)
            if DG.check_pruning_group(group):
                group.prune()
        print(group)
        for i, (dep, idxs) in enumerate(group):
            if isinstance(dep.target.module, torch.nn.Conv2d) and 'out_channels' in str(dep.handler):
                print(str(dep.handler))
                print(dep.target._name)
                idx_name = names.index(dep.target._name)
                print(idx_name)
                mask = torch.tensor([(1 if i not in prune_idx else 0) for i in range(filter_ranks_to_use[idx_name].size()[0])]).type('torch.BoolTensor')
                print(mask)
                print(filter_ranks_to_use[idx_name].size())
                filter_ranks_to_use[idx_name] = torch.masked_select(filter_ranks_to_use[idx_name], mask)
                print(filter_ranks_to_use[idx_name].size())
        idx_conv += 1

    if num_conv_layers_pruned % 3 == 0 and num_conv_layers_pruned != 0 and train_model:
        print('Starting Training Cycle')
        pms.summary(model, torch.zeros((1, 3, 150,150)).to(device), show_input=False, print_summary=True, max_depth=5, show_parent_layers=True)
        model = train_network(model, device=device, epochs=100)
        print('Ended Training Cycle')
        train_model=False

test_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='test', data_transforms=transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
pms.summary(model, torch.zeros((1, 3, 150,150)).to(device), show_input=False, print_summary=True, max_depth=5, show_parent_layers=True)
#Train and testing
model.train()
model.to(device)
num_epochs = 100
model = train_network(model, device=device, epochs=num_epochs)
test_loss = torch.nn.CrossEntropyLoss()
model.eval()
train_test.test_network(model=model, testloader=testloader, loss_function=test_loss, device=device)