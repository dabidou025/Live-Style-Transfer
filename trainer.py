from re import S
import torch
import random
from tqdm import tqdm

class Trainer:
    def __init__(self, st_model, features_model, layers, content_layers, style_layers, style_gram_matrices, device):
        self.st_model = st_model
        self.features_model = features_model
        self.layers = layers
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.style_gram_matrices = style_gram_matrices
        self.device = device
        
        self.n_styles = len(style_gram_matrices)

        # for batch normalization
        self.batch_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1).to(device)
        self.batch_std = torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1).to(device)

    def normalize_batch(self, batch):
        return (batch - self.batch_mean) / self.batch_std

    def train(self, dataloader, optimizer, n_epochs, content_factor, style_factor, save_dir=None, save_step=1):

        MSE = torch.nn.MSELoss()

        self.st_model.train()

        content_losses = []
        style_losses = []

        pbar = tqdm(range(n_epochs))
        for epoch in pbar:

            content_loss_epoch = 0
            style_loss_epoch = 0
            for i_batch, img in enumerate(dataloader):
                img = img.to(self.device)
                optimizer.zero_grad()
    
                img_features = self.features_model.get_features(img)

                s = random.randint(0, self.n_styles-1)

                gen_img = self.st_model(img, s)
                gen_img = self.normalize_batch(gen_img)
                gen_features = self.features_model.get_features(gen_img)

                content_loss = 0
                for i in self.content_layers:
                    content_loss += MSE(img_features[i], gen_features[i])

                style_loss = 0
                for i in self.style_layers:
                    gen_gram_matrix = self.features_model.gram_batch_matrix(gen_features[i])
                    style_gram_matrix = self.style_gram_matrices[s][i]

                    style_loss += (1/len(self.style_layers)) * MSE(gen_gram_matrix, torch.cat(img.shape[0]*[style_gram_matrix]))

                total_loss = content_factor*content_loss + style_factor*style_loss

                pbar.set_postfix({'progress': '{0:.2f}'.format(100*i_batch/len(dataloader)) + '%'})

                total_loss.backward()
                optimizer.step()

                content_loss_epoch += content_loss.item()
                style_loss_epoch += style_loss.item()
            
            pbar.set_description('{:.5e}'.format(content_loss_epoch) + ", " + '{:.5e}'.format(style_loss_epoch))

            content_losses.append(content_loss_epoch)
            style_losses.append(style_loss_epoch)

            if epoch % save_step == 0 and save_dir != None:
                torch.save(self.st_model.state_dict(), save_dir + '/st_model_' + str(epoch) + '.pth')

            
        if save_dir != None:
            torch.save(self.st_model.state_dict(), save_dir + '/st_model_FINAL.pth')

        return content_losses, style_losses


                
