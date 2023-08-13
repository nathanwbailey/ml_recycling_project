import torch
import torchvision

class RecycleNetwork(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.backbone = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
        self.backbone.classifier[3] = torch.nn.Linear(1024, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

