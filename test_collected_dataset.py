import torch
import torchvision
import sys
import dataset_collected
import network
from PIL import Image


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mean = torch.Tensor([0.6661, 0.6211, 0.5492])
std = torch.Tensor([0.2871, 0.2917, 0.3310])

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])


model = network.RecycleNetwork(num_classes=5).to(device)
model.load_state_dict(torch.load('recycle_net_trained.pt'))
model.eval()

dataset = dataset_collected.RecyclingDataset(data_dir='collected_dataset_with_labels', dataset_type='train', data_transforms=transforms)

testloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)


with torch.no_grad():
    for batch in testloader:
        images = batch[0].to(device)
        labels = batch[1].to(device)
        output = model(images)
        softmaxed_output = torch.nn.functional.softmax(output, dim=1)
        predictions = torch.argmax(softmaxed_output, dim=1)
        print(labels)
        print(predictions)