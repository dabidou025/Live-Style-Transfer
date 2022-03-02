import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np
import math
import time

class VGG16():
    def __init__(self, layers, device):
        super(VGG16,self).__init__()

        self.layers = layers

        self.vgg = models.vgg16(pretrained=True).features
        self.vgg = self.vgg.to(device)

        for param in self.vgg.parameters():
            param.requires_grad_(False)
    
    def get_features(self, x):
        features = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features[i] = x
        
        return features

    def gram_batch_matrix(self, x):
        batch_size, n_feature_maps, height, width = x.size()
        x = x.view(batch_size, n_feature_maps, height*width)
        gram = torch.bmm(x, x.transpose(1,2)) / (n_feature_maps * height * width) 

        return gram