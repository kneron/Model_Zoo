import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
import time

class mobile_net_v2(nn.Module):
    def __init__(self, num_classes=2):
        super(mobile_net_v2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        # replace the last FC layer by a FC layer for our model
        #num_ftrs = self.mobile_model.classifier.in_features
        num_ftrs =  self.model.classifier[-1].in_features
        #self.mobile_model.reset_classifier(0)
        self.model.classifier[1] = nn.Linear(num_ftrs//4*3, num_classes, bias=True)
        
        self.model.features[0][0] = nn.Conv2d(3, 32//4*3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False)
        self.model.features[0][1] = nn.BatchNorm2d(32//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[1].conv[0][0] = nn.Conv2d(32//4*3, 32//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32//4*3, bias=False)
        self.model.features[1].conv[0][1] = nn.BatchNorm2d(32//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[1].conv[1] = nn.Conv2d(32//4*3, 16//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[1].conv[2] = nn.BatchNorm2d(16//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[2].conv[0][0] = nn.Conv2d(16//4*3, 96//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[2].conv[0][1] = nn.BatchNorm2d(96//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[2].conv[1][0] = nn.Conv2d(96//4*3, 96//4*3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96//4*3, bias=False)
        self.model.features[2].conv[1][1] = nn.BatchNorm2d(96//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[2].conv[2] = nn.Conv2d(96//4*3, 24//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[2].conv[3] = nn.BatchNorm2d(24//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.model.features[3].conv[0][0] = nn.Conv2d(24//4*3, 128//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[3].conv[0][1] = nn.BatchNorm2d(128//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[3].conv[1][0] = nn.Conv2d(128//4*3, 128//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128//4*3, bias=False)
        self.model.features[3].conv[1][1] = nn.BatchNorm2d(128//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[3].conv[2] = nn.Conv2d(128//4*3, 24//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[3].conv[3] = nn.BatchNorm2d(24//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[4].conv[0][0] = nn.Conv2d(24//4*3, 144//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[4].conv[0][1] = nn.BatchNorm2d(144//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[4].conv[1][0] = nn.Conv2d(144//4*3, 144//4*3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144//4*3, bias=False)
        self.model.features[4].conv[1][1] = nn.BatchNorm2d(144//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[4].conv[2] = nn.Conv2d(144//4*3, 32//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[4].conv[3] = nn.BatchNorm2d(32//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[5].conv[0][0] = nn.Conv2d(32//4*3, 176//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[5].conv[0][1] = nn.BatchNorm2d(176//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[5].conv[1][0] = nn.Conv2d(176//4*3, 176//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=176//4*3, bias=False)
        self.model.features[5].conv[1][1] = nn.BatchNorm2d(176//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[5].conv[2] = nn.Conv2d(176//4*3, 32//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[5].conv[3] = nn.BatchNorm2d(32//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[6].conv[0][0] = nn.Conv2d(32//4*3, 192//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[6].conv[0][1] = nn.BatchNorm2d(192//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[6].conv[1][0] = nn.Conv2d(192//4*3, 192//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192//4*3, bias=False)
        self.model.features[6].conv[1][1] = nn.BatchNorm2d(192//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[6].conv[2] = nn.Conv2d(192//4*3, 32//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[6].conv[3] = nn.BatchNorm2d(32//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[7].conv[0][0] = nn.Conv2d(32//4*3, 192//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[7].conv[0][1] = nn.BatchNorm2d(192//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[7].conv[1][0] = nn.Conv2d(192//4*3, 192//4*3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192//4*3, bias=False)
        self.model.features[7].conv[1][1] = nn.BatchNorm2d(192//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[7].conv[2] = nn.Conv2d(192//4*3, 64//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[7].conv[3] = nn.BatchNorm2d(64//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[8].conv[0][0] = nn.Conv2d(64//4*3, 368//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[8].conv[0][1] = nn.BatchNorm2d(368//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[8].conv[1][0] = nn.Conv2d(368//4*3, 368//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=368//4*3, bias=False)
        self.model.features[8].conv[1][1] = nn.BatchNorm2d(368//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[8].conv[2] = nn.Conv2d(368//4*3, 64//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[8].conv[3] = nn.BatchNorm2d(64//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[9].conv[0][0] = nn.Conv2d(64//4*3, 384//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[9].conv[0][1] = nn.BatchNorm2d(384//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[9].conv[1][0] = nn.Conv2d(384//4*3, 384//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384//4*3, bias=False)
        self.model.features[9].conv[1][1] = nn.BatchNorm2d(384//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[9].conv[2] = nn.Conv2d(384//4*3, 64//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[9].conv[3] = nn.BatchNorm2d(64//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[10].conv[0][0] = nn.Conv2d(64//4*3, 384//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[10].conv[0][1] = nn.BatchNorm2d(384//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[10].conv[1][0] = nn.Conv2d(384//4*3, 384//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384//4*3, bias=False)
        self.model.features[10].conv[1][1] = nn.BatchNorm2d(384//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[10].conv[2] = nn.Conv2d(384//4*3, 64//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[10].conv[3] = nn.BatchNorm2d(64//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[11].conv[0][0] = nn.Conv2d(64//4*3, 384//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[11].conv[0][1] = nn.BatchNorm2d(384//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[11].conv[1][0] = nn.Conv2d(384//4*3, 384//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384//4*3, bias=False)
        self.model.features[11].conv[1][1] = nn.BatchNorm2d(384//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[11].conv[2] = nn.Conv2d(384//4*3, 96//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[11].conv[3] = nn.BatchNorm2d(96//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[12].conv[0][0] = nn.Conv2d(96//4*3, 560//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[12].conv[0][1] = nn.BatchNorm2d(560//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[12].conv[1][0] = nn.Conv2d(560//4*3, 560//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=560//4*3, bias=False)
        self.model.features[12].conv[1][1] = nn.BatchNorm2d(560//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[12].conv[2] = nn.Conv2d(560//4*3, 96//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[12].conv[3] = nn.BatchNorm2d(96//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[13].conv[0][0] = nn.Conv2d(96//4*3, 576//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[13].conv[0][1] = nn.BatchNorm2d(576//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[13].conv[1][0] = nn.Conv2d(576//4*3, 576//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576//4*3, bias=False)
        self.model.features[13].conv[1][1] = nn.BatchNorm2d(576//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[13].conv[2] = nn.Conv2d(576//4*3, 96//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[13].conv[3] = nn.BatchNorm2d(96//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[14].conv[0][0] = nn.Conv2d(96//4*3, 576//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[14].conv[0][1] = nn.BatchNorm2d(576//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[14].conv[1][0] = nn.Conv2d(576//4*3, 576//4*3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576//4*3, bias=False)
        self.model.features[14].conv[1][1] = nn.BatchNorm2d(576//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[14].conv[2] = nn.Conv2d(576//4*3, 160//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[14].conv[3] = nn.BatchNorm2d(160//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[15].conv[0][0] = nn.Conv2d(160//4*3, 960//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[15].conv[0][1] = nn.BatchNorm2d(960//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[15].conv[1][0] = nn.Conv2d(960//4*3, 960//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960//4*3, bias=False)
        self.model.features[15].conv[1][1] = nn.BatchNorm2d(960//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[15].conv[2] = nn.Conv2d(960//4*3, 160//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[15].conv[3] = nn.BatchNorm2d(160//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[16].conv[0][0] = nn.Conv2d(160//4*3, 960//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[16].conv[0][1] = nn.BatchNorm2d(960//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[16].conv[1][0] = nn.Conv2d(960//4*3, 960//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960//4*3, bias=False)
        self.model.features[16].conv[1][1] = nn.BatchNorm2d(960//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[16].conv[2] = nn.Conv2d(960//4*3, 160//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[16].conv[3] = nn.BatchNorm2d(160//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[17].conv[0][0] = nn.Conv2d(160//4*3, 960//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[17].conv[0][1] = nn.BatchNorm2d(960//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[17].conv[1][0] = nn.Conv2d(960//4*3, 960//4*3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960//4*3, bias=False)
        self.model.features[17].conv[1][1] = nn.BatchNorm2d(960//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.features[17].conv[2] = nn.Conv2d(960//4*3, 320//4*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.features[17].conv[3] = nn.BatchNorm2d(320//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.model.features[18][0] = nn.Conv2d(320//4*3, 1280//4*3, kernel_size=(1, 1), stride=(1, 1),bias=False)
        self.model.features[18][1] = nn.BatchNorm2d(1280//4*3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
    def forward(self, x):
        f = self.model(x)
        #y = self.classifier(f)
        return f
