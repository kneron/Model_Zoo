import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

class resnet18(nn.Module):
    def __init__(self, num_classes):
        super(resnet18, self).__init__()
        self.model =  models.resnet18(pretrained=False)
        # replace the last FC layer by a FC layer for our model
        num_ftrs =  self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes, bias=True)
        nn.init.xavier_uniform_(self.model.fc.weight)
        self.model.fc.bias.data.fill_(0.01)
        
    def forward(self, x):
        f = self.model(x)
        return f
