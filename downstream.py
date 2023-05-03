import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pytorch_lightning as pl

from models import AutoEncoder, AutoEncoder_VGG, AutoEncoder_ResNet

# import wandb
# wandb.login()

from torchmetrics import Accuracy

import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

class DownstreamWrapper(pl.LightningModule):
    def __init__(self, autoencoder_model, num_classes, features_dim):
        super().__init__()
        self.autoencoder_model = autoencoder_model
        # self.encoder_model = autoencoder_model.encoder
        self.hidden = torch.nn.Sequential(
            # torch.nn.Linear(features_dim, 256),
            # torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout(p=0.1),
            torch.nn.Linear(features_dim, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.05)
        )
        self.classifier = torch.nn.Linear(128, num_classes)
        
        self.training_losses = []
        self.training_accs = []
        self.validation_losses = []
        self.validation_accs = []
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        self.accuracy = Accuracy('multiclass', num_classes = num_classes)

    def forward(self, x):
        with torch.no_grad():
            features, _ = self.autoencoder_model(x)
        hidden = self.hidden(features)
        output = self.classifier(hidden)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        # self.log('train_loss', loss, prog_bar=True)
        # self.log('train_acc', acc, prog_bar=True)
        
        
        self.training_losses.append(loss)
        self.training_accs.append(acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        # self.log('val_loss', loss, prog_bar=True)
        # self.log('val_acc', acc, prog_bar=True)
        self.validation_losses.append(loss)
        self.validation_accs.append(acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=8)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_loss'}
        
    def on_train_epoch_end(self):
        loss_avg = torch.stack(self.training_losses).mean()
        acc_avg = torch.stack(self.training_accs).mean()
        self.log("train_loss", loss_avg, prog_bar=True)
        self.log("train_acc", acc_avg, prog_bar=True)
        self.training_losses.clear()  # free memory
        self.training_accs.clear()  # free memory

    def on_validation_epoch_end(self):
        loss_avg = torch.stack(self.validation_losses).mean()
        acc_avg = torch.stack(self.validation_accs).mean()
        self.log("valid_loss", loss_avg, prog_bar=True)
        self.log("valid_acc", acc_avg, prog_bar=True)
        self.validation_losses.clear()  # free memory
        self.validation_accs.clear()  # free memory
        

# Define transforms for data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
])

# Define dataset
dataset = ImageFolder('train', transform=train_transform)

# Define train and validation datasets
val_size = int(len(dataset) * 0.15)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Use different transforms for train and val loaders
train_dataset.transforms = train_transform
val_dataset.transforms = val_transform





# autoencoder_model = AutoEncoder(latent_dim=256)
# autoencoder_model = AutoEncoder_VGG(latent_dim=256)
autoencoder_model = AutoEncoder_ResNet(latent_dim=1024)

autoencoder_model_name = 'AUTOENCODER_PRETRAIN_RESNET_PSE'
if os.path.exists("models/"+autoencoder_model_name+".pt"):
    # load the model
    checkpoint = torch.load("models/"+autoencoder_model_name+".pt")
    autoencoder_model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using Device", device)

# from pytorch_lightning.loggers import WandbLogger
# wandb_logger = WandbLogger(project='tudelft_interview')
early_stop_callback = pl.callbacks.EarlyStopping(monitor="valid_loss" , patience=50)
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="models/"+autoencoder_model_name, save_top_k=1, monitor="valid_loss")


model = DownstreamWrapper(autoencoder_model, num_classes=10, features_dim=1024)
model.to(device)
trainer = pl.Trainer(max_epochs=500, gradient_clip_val=0.5, callbacks=[early_stop_callback,
                                                                    checkpoint_callback])
trainer.fit(model, train_loader, val_loader)