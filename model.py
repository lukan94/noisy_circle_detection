import torch.nn as nn
import numpy as np



class NoisyCircle_Detector(nn.Module):
    def __init__(self):
        super(NoisyCircle_Detector, self).__init__()
        self.layer1 = self._make_layer(3,64)
        self.layer2 = self._make_layer(64,64)
        self.layer3 = self._make_layer(64,128)
        self.layer4 = self._make_layer(128,128)
        self.layer5 = self._make_layer(128,128)
        self.layer6 = self._make_layer(128,256)
        self.linear = nn.Linear(2304, 3)       
 
    def _make_layer(self, in_channels, out_channels):
        layer = []
        layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layer.append(nn.BatchNorm2d(out_channels, affine = True))
        layer.append(nn.ReLU(inplace=True))
        layer.append(nn.MaxPool2d(2))

        return nn.Sequential(*layer)
 
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x
