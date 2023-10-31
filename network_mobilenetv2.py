import torch
import torchvision
import pytorch_model_summary as pms

class RecycleNetwork(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.backbone = torchvision.models.mobilenet_v2(weights='DEFAULT')
        self.backbone.classifier[1] = torch.nn.Linear(1280, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


model = RecycleNetwork(5)
pms.summary(model, torch.zeros((1, 3, 150,150)), show_input=False, print_summary=True, max_depth=5, show_parent_layers=True)