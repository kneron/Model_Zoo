import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms


class resnet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(resnet50, self).__init__()
        self.model =  models.resnet50(pretrained=False)
        # replace the last FC layer by a FC layer for our model
        num_ftrs =  self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes, bias=True)
               
    def forward(self, x):
        f = self.model(x)
        return f
