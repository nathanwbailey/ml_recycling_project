import torch
import torchvision
import sys
sys.path.append('./EfficientNet')
from efficientnet_lite import *
import pytorch_model_summary as pms



class RecycleNetwork(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        model_name = 'efficientnet_lite4'
        self.backbone = build_efficientnet_lite(model_name, 1000)
        self.backbone.load_pretrain('./EfficientNet/efficientnet_lite4.pth')
        self.backbone.fc = torch.nn.Linear(1280, 5)

    def forward(self, x):
        return self.backbone(x)

model = RecycleNetwork(5)
print(model)
#pms.summary(model, torch.zeros((1, 3, 150,150)), show_input=False, print_summary=True, max_depth=5, show_parent_layers=True)
