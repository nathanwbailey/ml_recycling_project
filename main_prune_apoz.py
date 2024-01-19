import torch
import torchvision
import sys
import dataset
import network_apoz
import dataset_apoz
import train_test
import torch_pruning as pruning
import math
import pytorch_model_summary as pms
import apoz_prune
import pickle


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

mean = torch.Tensor([0.6661, 0.6211, 0.5492]).to('cpu')
std = torch.Tensor([0.2871, 0.2917, 0.3310]).to('cpu')

model = network_apoz.RecycleNetwork(num_classes=5).to(device)
model.load_state_dict(torch.load('recycle_net_trained.pt'))
model.eval()

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

valid_dataset = dataset_apoz.RecyclingDataset(data_dir='compiled_dataset', dataset_type='valid', data_transforms=transforms)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
print(len(validloader))

layer_names = [name for name, _ in model.named_modules()]
layer_names = [name for name in layer_names.copy() if 'stem.0' in name or 'head.0' in name or 'project_conv' in name or 'expand_conv' in name or 'depthwise_conv' in name or 'se_reduce' in name or 'se_expand' in name]
# print(layer_names)
outputs_conv_blocks = None
outputs_conv_stem = None
outputs_conv_head = None
with torch.no_grad():
    batch = next(iter(validloader))
    images = batch[0].to(device)
    outputs, _ = model(images)
    inter_outputs = [i.cpu() for i in outputs[0] if i is not None]
    for i, item in enumerate(inter_outputs):
        if item is not None:
            globals()['outputs_inner_layer_'+str(i)] = None

collected_outputs = []
masks = []

prune_percentage = 0.95
with torch.no_grad():
    for idx, batch in enumerate(validloader):
        images = batch[0].to(device)
        print(batch[1])
        outputs, _ = model(images)
        inter_outputs = [i.cpu() for i in outputs[0] if i is not None]
        if idx == 0:
            outputs_conv_stem = outputs[1].cpu()
            outputs_conv_head = outputs[2].cpu()
            for i, item in enumerate(inter_outputs):
                if item is not None:
                    globals()['outputs_inner_layer_'+str(i)] = item
        else:
            outputs_conv_stem = torch.cat((outputs_conv_stem, outputs[1].cpu()), dim=0)
            outputs_conv_head = torch.cat((outputs_conv_head, outputs[2].cpu()), dim=0)
            for i, item in enumerate(inter_outputs):
                if item is not None:
                    globals()['outputs_inner_layer_'+str(i)] = torch.cat((globals()['outputs_inner_layer_'+str(i)], item), dim=0)
    collected_outputs.append(outputs_conv_stem)

    masks.append(apoz_prune.compute_mask_conv_layer(outputs_conv_stem, prune_percentage))
    for i, item in enumerate(inter_outputs):
        if item is not None:
            collected_outputs.append(globals()['outputs_inner_layer_'+str(i)])
            # print(apoz_prune.apoz_per_feature_map(globals()['outputs_inner_layer_'+str(i)]))
            # print(apoz_prune.compute_mask_conv_layer(globals()['outputs_inner_layer_'+str(i)], prune_percentage))
            prune_idx = [i for i, j in enumerate(apoz_prune.compute_mask_conv_layer(globals()['outputs_inner_layer_'+str(i)], prune_percentage)) if j == 1]
            # print(prune_idx)
            # print(len(prune_idx))
            masks.append(apoz_prune.compute_mask_conv_layer(globals()['outputs_inner_layer_'+str(i)], prune_percentage))
    collected_outputs.append(outputs_conv_head)
    masks.append(apoz_prune.compute_mask_conv_layer(outputs_conv_head, prune_percentage))

collected_outputs.reverse()
masks.reverse()
layer_names.reverse()


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

    model = train_test.train_network(model=model, num_epochs=epochs, optimizer=optimizer, loss_function=loss, trainloader=trainloader, validloader=validloader, device=device, scheduler=scheduler)
    return model

def prune_apoz_train(model, idx_to_prune, num, epochs):
    model.eval()
    model.to(device)
    conv_idx = apoz_prune.compute_mask_conv_layer(collected_outputs[idx_to_prune], num).nonzero()
    prune_idx = conv_idx.reshape(conv_idx.size(0)).tolist()
    with open('idx_'+str(idx_to_prune)+'.pkl', 'wb') as f:
        pickle.dump(prune_idx, f)
    named_layer = layer_names[idx_to_prune]
    layer_to_use = None
    for name, layer in model.named_modules():
        if name == named_layer:
            layer_to_use = layer
    DG = pruning.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1,3,150,150).to(device))
    if layer_to_use.groups == layer_to_use.in_channels:
        print('Depthwise Detected')
        group = DG.get_pruning_group(layer_to_use, pruning_fn=pruning.prune_depthwise_conv_out_channels, idxs=prune_idx)
        if DG.check_pruning_group(group):
            group.prune()
    else:
        group = DG.get_pruning_group(layer_to_use, pruning_fn=pruning.prune_conv_out_channels, idxs=prune_idx)
        if DG.check_pruning_group(group):
            group.prune()
    print(group)
    model.to(device)
    # pms.summary(model, torch.zeros((1, 3, 150,150)).to(device), show_input=False, print_summary=True, max_depth=5, show_parent_layers=True)
    model = train_network(model, device, epochs)
    return model


model = network.RecycleNetwork(num_classes=5).to(device)
model.load_state_dict(torch.load('recycle_net_trained.pt'))
model.train()

#11 prunes

DG = pruning.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,150,150).to(device))
model.to(device)
num_conv_layers_pruned = 0
train_model=True
idx_conv = 0
num_prunes = 0

num_pruned = 0


for idx, (name,layer) in enumerate(model.named_modules()):
    if isinstance(layer, torch.nn.Conv2d):
        num_conv_layers_pruned +=1
        train_model=True
        prune_idx = masks[len(masks)-idx_conv-1]
        # print(prune_idx.size())
        prune_idx = [i for i, j in enumerate(prune_idx) if j == 1]
        name = layer_names[len(layer_names)-idx_conv-1]
        print(name)
        # print(prune_idx)
        print(len(prune_idx))
        num_pruned += len(prune_idx)
        # print(layer.weight.size())
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
                idx_name = layer_names.index(dep.target._name)
                print(idx_name)
                mask = torch.tensor([(1 if i not in prune_idx else 0) for i in range(masks[idx_name].size()[0])]).type('torch.BoolTensor')
                print(mask)
                print(masks[idx_name].size())
                masks[idx_name] = torch.masked_select(masks[idx_name], mask)
                print(masks[idx_name].size())
        idx_conv += 1
    if num_conv_layers_pruned % 3 == 0 and num_conv_layers_pruned != 0 and train_model:
        print('Starting Training Cycle')
        pms.summary(model, torch.zeros((1, 3, 150,150)).to(device), show_input=False, print_summary=True, max_depth=5, show_parent_layers=True)
        #model = train_network(model, device=device, epochs=100)
        print('Ended Training Cycle')
        train_model=False

print(num_pruned)

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