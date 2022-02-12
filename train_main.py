import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import time
import os
import copy
import pandas as pd
from PIL import Image
from skimage import io, transform
import random
from tqdm import tqdm
from sklearn.utils import shuffle
from random import randint

from picture_dataset import PictureDataset
from models.vgg16 import VGG16
from models.stmodel import STModel
from trainer import Trainer

def main():
    device = torch.device("cpu")

    layers = [3,8,15,22]
    content_layers = [15]
    style_layers = [3,8,15,22]

    n_styles = 10
    st_model = STModel(n_styles)
    st_model = st_model.to(device)

    features_model = VGG16(layers, device)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #trainer = Trainer(st_model, features_model, layers, content_layers, style_layers, style_gram_matrices, device)
    
    img_size = 256
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    dataset = PictureDataset('', size=10, transform=data_transform)
    print(dataset.list_images)

    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    #optimizer = torch.optim.Adam(st_model.parameters(), lr=5e-4, betas=(0.9, 0.999))
    
    #content_losses, style_losses = trainer.train(dataloader, optimizer, n_epochs=2, content_factor=1, style_factor=1e4, save_dir='models')

if __name__ == '__main__':
    main()