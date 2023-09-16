import torch
import torchvision
import sys
import dataset
from network_apoz import RecycleNetwork
import train_test
from apoz_prune import *
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

mean = torch.Tensor([0.6661, 0.6211, 0.5492]).to('cpu')
std = torch.Tensor([0.2871, 0.2917, 0.3310]).to('cpu')

model = RecycleNetwork(num_classes=5).to(device)

model.load_state_dict(torch.load('recycle_net_trained.pt'))
model.eval()

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

valid_dataset = dataset.RecyclingDataset(data_dir='compiled_dataset', dataset_type='valid', data_transforms=transforms)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

outputs_conv_blocks = None
outputs_conv_stem = None
outputs_conv_head = None

with torch.no_grad():
    for idx, batch in enumerate(validloader):
        images = batch[0].to(device)
        outputs, _ = model(images)
        if idx == 0:
            outputs_conv_stem = outputs[1]
            outputs_conv_head = outputs[2]
            #print(outputs[0])
            inter_outputs = [i.cpu() for i in outputs[0]]
            inter_outputs_mask = (np.array(inter_outputs) == None)
            print(inter_outputs_mask)
            inter_outputs = outputs[0](outputs[0] == None)
            print(inter_outputs)
            outputs_conv_blocks = torch.tensor(outputs[0])
            print(outputs_conv_blocks.size())
        else:
            outputs_conv_stem = torch.cat((outputs_conv_stem, outputs[1]), dim=0)
            outputs_conv_head = torch.cat((outputs_conv_head, outputs[1]), dim=0)


