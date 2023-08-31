from torch import nn
import torch


class DetectionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding='same'),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(1,1))
        )
        self.classify = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.Linear(in_features=64, out_features=8),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=8, out_features=2)            
        )
    def forward(self, x):
        x = self.features(x)
        print(x.size())
        x = x.reshape(x.size(0), -1)
        x = self.classify(x)
        return x


#Test the Network
# network = DetectionNetwork()
# network.eval()
# input_test = torch.rand(5, 3, 150, 150)
# print(network(input_test).size())
