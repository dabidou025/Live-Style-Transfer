import torch
from torchvision import transforms

from PIL import Image

import numpy as np

class Predictor:
    def __init__(self, st_model, device, img_size):
        self.device = device

        self.st_model = st_model.to(device)
        self.st_model.eval()
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]   
        
        self.transformer = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
    def eval_image(self, img, style_1, style_2=None, alpha=0.5):
        img = self.transformer(img).to(self.device)
        gen = self.st_model(img.unsqueeze(0), style_1, style_2, alpha)
        
        return Image.fromarray(np.uint8(np.moveaxis(gen[0].cpu().detach().numpy()*255.0, 0, 2)))

class WebcamPredictor:
    def __init__(self, st_model, device):
        self.device = device

        self.st_model = st_model.to(device)
        self.st_model.eval()
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.mean = np.expand_dims(self.mean, (1,2)).to(device)
        self.std = np.expand_dims(self.std, (1,2)).to(device)

    def eval_image(self, img, style_1, style_2=None, alpha=0.5):
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).to(self.device)

        gen = self.st_model(img.unsqueeze(0), style_1, style_2, alpha)
        
        return np.uint8(img[0].cpu().detach().numpy()*255.0)