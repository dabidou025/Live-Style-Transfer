from torchvision import transforms
from PIL import Image
import numpy as np

class Predictor():
    def __init__(self, st_model, device, img_size):
        self.device = device

        self.st_model = st_model.to(device)
        st_model.eval()
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]   
        
        self.transformer = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
    def eval_image(self, img, style_1, style_2=None, alpha=0.5):
        img = self.transformer(img)
        gen = self.st_model(img.unsqueeze(0).to(self.device), style_1, style_2, alpha)
        
        return Image.fromarray(np.uint8(np.moveaxis(gen[0].cpu().detach().numpy()*255.0, 0, 2)))