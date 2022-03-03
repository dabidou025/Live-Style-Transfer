from torch.utils.data import Dataset

import random

from PIL import Image
from glob import glob
import os

class PictureDataset(Dataset):
    def __init__(self, root_dir, size=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.list_images = glob(os.path.join(self.root_dir, '*.jpg'))
        if size != None:
            random.shuffle(self.list_images)
            self.list_images = self.list_images[:size]

                                
    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        image_loc = self.list_images[idx]
        image = Image.open(image_loc).convert('RGB')
        
        if self.transform != None:
            image = self.transform(image)
        
        return image