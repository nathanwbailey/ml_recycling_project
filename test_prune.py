import torch
from torchvision.models import resnet18
import torch_pruning as tp

model = resnet18(pretrained=True).eval()
print(model)

# 1. Build dependency graph for resnet18
DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 2. Group all coupled layers
group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )

# 3. Prune grouped layers altogether
if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.
    group.prune()

print(group.details())