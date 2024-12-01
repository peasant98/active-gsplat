
import torch
import torch.nn as nn
from torchvision import models

class ResNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(ResNetBinaryClassifier, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    ## Example loading the model with saved checkpoint
    model = ResNetBinaryClassifier()
    model.load_state_dict(torch.load("../models/resnet/kitchen_resnet.pth")) #replace with location of .pth checkpoint