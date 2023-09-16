import torch
import torchvision
import sys
import dataset
import network
from PIL import Image


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mean = torch.Tensor([0.6661, 0.6211, 0.5492])
std = torch.Tensor([0.2871, 0.2917, 0.3310])

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])


model = network.RecycleNetwork(num_classes=5).to(device)
model.load_state_dict(torch.load('recycle_net.pt'))
model.eval()



image = transforms(Image.open('test_image.jpg')).to(device)
print(type(image))
image = image[None, ...]
print(image.size())

output = model(image)
softmaxed_output = torch.nn.functional.softmax(output, dim=1)

print(softmaxed_output)
prediction = torch.argmax(softmaxed_output, dim=1)
print(prediction)