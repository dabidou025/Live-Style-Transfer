import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from picture_dataset import PictureDataset
from models.stmodel import STModel
from models.vgg16 import VGG16
from trainer import Trainer

import argparse

def train(args):

    dataset_path = args.dataset_path
    styles_path = args.styles_path
    save_model_path = args.save_model_path
    n_epochs = args.n_epochs

    dataset_size = args.dataset_size

    batch_size = args.batch_size
    lr = args.lr

    content_factor = args.content_factor
    style_factor = args.style_factor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layers = [3,8,15,22]
    content_layers = [22]
    style_layers = [3,8,15,22]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_size = 64
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    style_dataset = PictureDataset(styles_path, transform=data_transform)
    n_styles = len(style_dataset)
    st_model = STModel(n_styles)
    if False:
        st_model.load_state_dict(torch.load('D:/code/models/st_model_FINAL.pth'))
    st_model = st_model.to(device)

    features_model = VGG16(layers, device)

    style_gram_matrices = []
    for s in range(len(style_dataset)):
        style_matrices = {}
        style_img = style_dataset[s].unsqueeze(0).to(device)
        style_features = features_model.get_features(style_img)
        for i in style_layers:
            style_matrices[i] = features_model.gram_batch_matrix(style_features[i]).to(device)
        style_gram_matrices.append(style_matrices)
    
    trainer = Trainer(st_model, features_model, layers, content_layers, style_layers, style_gram_matrices, device)
    
    dataset = PictureDataset(dataset_path, size=dataset_size, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(st_model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    content_losses, style_losses = trainer.train(dataloader, optimizer, n_epochs=n_epochs, content_factor=content_factor, style_factor=style_factor, save_dir=save_model_path)

    plt.subplot(1, 2, 1)
    plt.plot(content_losses)
    plt.subplot(1, 2, 2)
    plt.plot(style_losses)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="parser for training mutli-style-transfer")
    
    parser.add_argument("--n-epochs", type=int, default=2)

    parser.add_argument("--batch-size", type=int, default=1)
    
    parser.add_argument("--dataset-path", type=str, required=True)

    parser.add_argument("--styles-path", type=str, required=True)

    parser.add_argument("--dataset-size", type=int, required=True)

    parser.add_argument("--lr", type=float, default=1e-3)
    
    parser.add_argument("--content-factor", type=int, default=1)

    parser.add_argument("--style-factor", type=int, default=1e6)

    parser.add_argument("--save-model-path", type=str, default=None)
    
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()