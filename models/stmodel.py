import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import time

class ConvCIN(nn.Module):
    def __init__(self, n_styles, C_in, C_out, kernel_size, padding, stride, activation=None):
        super(ConvCIN, self).__init__()
        
        self.reflection = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=kernel_size, stride=stride)
        nn.init.normal_(self.conv.weight, mean=0, std=1e-2)

        self.instnorm = nn.InstanceNorm2d(C_out)#, affine=True)
        #nn.init.normal_(self.instnorm.weight, mean=1, std=1e-2)
        #nn.init.normal_(self.instnorm.bias, mean=0, std=1e-2)

        
        self.gamma = torch.nn.Parameter(data=torch.randn(n_styles, C_out)*1e-2 + 1, requires_grad=True)
        #self.gamma.data.uniform_(1.0, 1.0)

        self.beta = torch.nn.Parameter(data=torch.randn(n_styles, C_out)*1e-2, requires_grad=True)
        #self.beta.data.uniform_(0, 0)

        self.activation = activation

    def forward(self, x, style_1, style_2, alpha):

        x = self.reflection(x)
        x = self.conv(x)

        x = self.instnorm(x)

        
        if style_2 != None:
            gamma = alpha*self.gamma[style_1] + (1-alpha)*self.gamma[style_2]
            beta = alpha*self.beta[style_1] + (1-alpha)*self.beta[style_2]
        else:
            gamma = self.gamma[style_1]
            beta = self.beta[style_1]
        

        b,d,w,h = x.size()
        x = x.view(b,d,w*h)

        x = (x*gamma.unsqueeze(-1) + beta.unsqueeze(-1)).view(b,d,w,h)

        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, n_styles, C_in, C_out):
        super(ResidualBlock,self).__init__()

        self.convcin1 = ConvCIN(n_styles, C_in, C_out, kernel_size=3, padding=1, stride=1, activation='relu')
        self.convcin2 = ConvCIN(n_styles, C_in, C_out, kernel_size=3, padding=1, stride=1)

    def forward(self, x, style_1, style_2, alpha):
        out = self.convcin1(x, style_1, style_2, alpha)
        out = self.convcin2(out, style_1, style_2, alpha)
        return x + out

class UpSampling(nn.Module):
    def __init__(self, n_styles, C_in, C_out):
        super(UpSampling,self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.convcin = ConvCIN(n_styles, C_in, C_out, kernel_size=3, padding=1, stride=1, activation='relu')

    def forward(self, x, style_1, style_2, alpha):
        x = self.upsample(x)
        x = self.convcin(x, style_1, style_2, alpha)
        return x

class STModel(nn.Module):
    def __init__(self, n_styles):
        super(STModel,self).__init__()

        self.convcin1 = ConvCIN(n_styles, C_in=3, C_out=32, kernel_size=9, padding=4, stride=1, activation='relu')
        self.convcin2 = ConvCIN(n_styles, C_in=32, C_out=64, kernel_size=3, padding=1, stride=2, activation='relu')
        self.convcin3 = ConvCIN(n_styles, C_in=64, C_out=128, kernel_size=3, padding=1, stride=2, activation='relu')

        self.rb1 = ResidualBlock(n_styles, 128, 128)
        self.rb2 = ResidualBlock(n_styles, 128, 128)
        self.rb3 = ResidualBlock(n_styles, 128, 128)
        self.rb4 = ResidualBlock(n_styles, 128, 128)
        self.rb5 = ResidualBlock(n_styles, 128, 128)

        self.upsample1 = UpSampling(n_styles, 128, 64)
        self.upsample2 = UpSampling(n_styles, 64, 32)

        self.convcin4 = ConvCIN(n_styles, C_in=32, C_out=3, kernel_size=9, padding=4, stride=1, activation='sigmoid')

    def forward(self, x, style_1, style_2=None, alpha=0.5):
        x = self.convcin1(x, style_1, style_2, alpha)
        x = self.convcin2(x, style_1, style_2, alpha)
        x = self.convcin3(x, style_1, style_2, alpha)

        x = self.rb1(x, style_1, style_2, alpha)
        x = self.rb2(x, style_1, style_2, alpha)
        x = self.rb3(x, style_1, style_2, alpha)
        x = self.rb4(x, style_1, style_2, alpha)
        x = self.rb5(x, style_1, style_2, alpha)

        x = self.upsample1(x, style_1, style_2, alpha)
        x = self.upsample2(x, style_1, style_2, alpha)

        x = self.convcin4(x, style_1, style_2, alpha)

        return x