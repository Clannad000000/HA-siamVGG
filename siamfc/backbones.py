import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import math
import numpy as np


class fuse_layer(nn.Module):
    def __init__(self):
        super(fuse_layer,self).__init__()
        self.fuse = nn.Sequential(nn.Conv2d(2,1,kernel_size=1,stride=1,padding=0))

    def forward(self,x):
        out = self.fuse(x)
        return out



class vgg_crop(nn.Module):
    def __init__(self, load_ini_weights = True,init_weights=True,k_size = 3):
        super(vgg_crop, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.layer7 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.layer8 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.layer9 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.layer10 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.layer11 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,padding=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.layer12 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        self.layer13 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True))

        self.layer14 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


        if init_weights:
                self._initialize_weights()

        if load_ini_weights:
                self._load_ini_weights()

    def crop(self,x):
        return x[:,:,1:-1,1:-1].contiguous()


    def forward(self, x,is_z = False):
        x = self.layer1(x)
        x = self.crop(x)
        x = self.layer2(x)
        x = self.crop(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x = self.crop(x)
        x = self.layer5(x)
        x = self.crop(x)
        x = self.layer6(x)

        x = self.layer7(x)
        x = self.crop(x)
        x = self.layer8(x)
        x = self.crop(x)
        x = self.layer9(x)
        x = self.crop(x)
        x3 = self.layer10(x)

        x = self.layer11(x3)
        x4 = self.crop(x)
        x = self.layer12(x4)
        x = self.crop(x)
        x = self.layer13(x)
        x = self.crop(x)

        x5 = self.layer14(x)

        if is_z:
            b5,c5,h5,w5 = x5.size()
            y5 = self.avg_pool(x5)# (8,256,1,1)
            y5 = self.conv(y5.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y5 = self.sigmoid(y5)#  (8,256,1,1)
            x5 = x5 * y5.expand_as(x5) + x5
        return x5,x4


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()


    def _load_ini_weights(self):
        pre_model = models.vgg16_bn(pretrained = True)
        m0 = list(pre_model.modules())
        m1 = list(self.modules())

        count = 0
        for m in self.modules():
            if count == 2:
                m.weight.data = m0[2].weight.data
            # if count == 3:
            #     m.weight.data = m0[3].weight.data
            #     m.bias.data = m0[3].bias.data
            #     m.running_mean = m0[3].running_mean
            #     m.running_var = m0[3].running_var
            if count == 6:
                m.weight.data = m0[5].weight.data
            # if count == 7:
            #     m.weight.data = m0[6].weight.data
            #     m.bias.data = m0[6].bias.data
            #     m.running_mean = m0[6].running_mean
            #     m.running_var = m0[6].running_var

            if count == 12:
                m.weight.data = m0[9].weight.data
            # if count == 13:
            #     m.weight.data = m0[10].weight.data
            #     m.bias.data = m0[10].bias.data
            #     m.running_mean = m0[10].running_mean
            #     m.running_var = m0[10].running_var
            if count == 16:
                m.weight.data = m0[12].weight.data
            # if count == 17:
            #     m.weight.data = m0[13].weight.data
            #     m.bias.data = m0[13].bias.data
            #     m.running_mean = m0[13].running_mean
            #     m.running_var = m0[13].running_var

            if count == 22:
                m.weight.data = m0[16].weight.data
            # if count == 23:
            #     m.weight.data = m0[17].weight.data
            #     m.bias.data = m0[17].bias.data
            #     m.running_mean = m0[17].running_mean
            #     m.running_var = m0[17].running_var
            if count == 26:
                m.weight.data = m0[19].weight.data
            # if count == 27:
            #     m.weight.data = m0[20].weight.data
            #     m.bias.data = m0[20].bias.data
            #     m.running_mean = m0[20].running_mean
            #     m.running_var = m0[20].running_var
            if count == 30:
                m.weight.data = m0[22].weight.data
            # if count == 31:
            #     m.weight.data = m0[23].weight.data
            #     m.bias.data = m0[23].bias.data
            #     m.running_mean = m0[23].running_mean
            #     m.running_var = m0[23].running_var

            if count == 36:
                m.weight.data = m0[26].weight.data
            # if count == 37:
            #     m.weight.data = m0[27].weight.data
            #     m.bias.data = m0[27].bias.data
            #     m.running_mean = m0[27].running_mean
            #     m.running_var = m0[27].running_var
            if count == 40:
                m.weight.data = m0[29].weight.data
            # if count == 41:
            #     m.weight.data = m0[30].weight.data
            #     m.bias.data = m0[30].bias.data
            #     m.running_mean = m0[30].running_mean
            #     m.running_var = m0[30].running_var
            if count == 44:
                m.weight.data = m0[32].weight.data
            # if count == 45:
            #     m.weight.data = m0[33].weight.data
            #     m.bias.data = m0[33].bias.data
            #     m.running_mean = m0[33].running_mean
            #     m.running_var = m0[33].running_var
            count += 1


if __name__ == "__main__":
    model = vgg_crop()
    x = Variable(torch.randn(1, 3, 255, 255))
    model(x)





