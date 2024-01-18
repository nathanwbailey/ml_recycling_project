import torch

def apoz_per_feature_map(output):
    return torch.sum((output == 0), dim = (0,2,3))/(output.size(0)*output.size(2)*output.size(3))

def compute_mask_conv_layer(output, val):
    return (apoz_per_feature_map(output) >= val).to(torch.uint8)

def apoz_per_neuron(output):
    return torch.sum((output == 0), dim=0)/output.size(0)

def compute_mask(output, val):
    return (apoz_per_neuron(output) >= val).to(torch.uint8)
