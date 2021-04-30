import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms


class mobilenet_v2(nn.Module):
    def __init__(self, num_classes=1000):
        super(mobilenet_v2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        # replace the last FC layer by a FC layer for our model
        num_ftrs =  self.model.classifier[-1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes, bias=True)
               
    def forward(self, x):
        f = self.model(x)
        return f
