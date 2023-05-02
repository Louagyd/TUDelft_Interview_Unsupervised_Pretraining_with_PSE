import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot, make_dot_from_trace

import wandb

from models import AutoEncoder, AutoEncoder_VGG, AutoEncoder_ResNet
from models_mae import mae_vit_large_patch16

from loss import PSELoss

import matplotlib.pyplot as plt

run = wandb.init(project='tudelft_interview')


# Define transforms for preprocessing and data augmentation
preprocess = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor()
])

augmentation = transforms.Compose([
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.RandomCrop(96, padding=4),
    transforms.RandomHorizontalFlip(p=0.1)
])

# Load the dataset from a folder and split into train and validation sets
dataset = datasets.ImageFolder('unlabeled', transform=preprocess)
train_size = int(0.85 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

# Create DataLoader objects for train and validation sets
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

# Add augmentation to the training DataLoader
train_loader.dataset.transform = transforms.Compose([
    augmentation,
    preprocess
])
    
# Define the model and optimizer

# model = AutoEncoder(latent_dim=256)
# model = AutoEncoder_VGG(latent_dim=256)
# model = AutoEncoder_ResNet(latent_dim=1024)
model = mae_vit_large_patch16(img_size=96)

model_name = "AUTOENCODER_PRETRAIN_MASKED_MSE"
if os.path.exists("models/"+model_name+".pt"):
    # load the model
    checkpoint = torch.load("models/"+model_name+".pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Define the loss function and device
MSELoss = nn.MSELoss()
PSELoss = PSELoss(3)

criterion = MSELoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using Device", device)
model.to(device)

# Define the TensorBoard writer
# writer = SummaryWriter('logs')
wandb.watch(model, log='all')

log_interval = 100
# Define the training loop
def train():
    model.train()
    running_mse = 0.0
    running_pse = 0.0
    running_grad_norm = 0.0
    
    for i, (inputs, _) in enumerate(train_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        latents, outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        mseloss = MSELoss(outputs, inputs)
        pseloss = PSELoss(outputs, inputs)
        
        running_mse += mseloss.item()
        running_pse += pseloss.item()
        running_grad_norm += grad_norm
        
        if i % log_interval == log_interval - 1:
            print("Iteration", i,
                  "MSE", running_mse / log_interval,
                  "PSE", running_pse / log_interval,
                  "GN", running_grad_norm / log_interval)
            wandb.log({"train/Loss/MSE": running_mse / log_interval,
                       "train/Loss/PSE": running_pse / log_interval,
                       "train/Gradient_Norm": running_grad_norm / log_interval})
            running_mse = 0.0
            running_pse = 0.0
            running_grad_norm = 0.0
        if i == 1:
            inputs_grid = wandb.Image(inputs[:16] , caption='Input Images')
            outputs_grid = wandb.Image(outputs[:16], caption='Output Images')
            mse_grid = wandb.Image(torch.abs(inputs[:16]-outputs[:16])**2, caption='MSE')
            pse_grid = wandb.Image(PSELoss.get_loss_image(inputs[:16],outputs[:16]), caption='PSE')
            wandb.log({'train/images/inputs': inputs_grid,
                       'train/images/outputs': outputs_grid,
                       'train/images/PSE': pse_grid,
                       'train/images/MSE': mse_grid})

            
# Define the validation loop
def validate():
    model.eval()
    running_mse = 0.0
    running_pse = 0.0
    ct = 0
    with torch.no_grad():
        for inputs, _ in valid_loader:
            inputs = inputs.to(device)
            latents, outputs = model(inputs)
            mseloss = MSELoss(outputs, inputs)
            pseloss = PSELoss(outputs, inputs)
            running_mse += mseloss.item()
            running_pse += pseloss.item()
            ct += 1
    
    wandb.log({"valid/Loss/MSE": running_mse / ct,
               "valid/Loss/PSE": running_pse / ct})
    
    rnd_ind = np.random.randint(8,58)
    ins = torch.concat([inputs[:8], inputs[rnd_ind:rnd_ind+8]])
    outs = torch.concat([outputs[:8], outputs[rnd_ind:rnd_ind+8]])
    inputs_grid = wandb.Image(ins , caption='Input Images')
    outputs_grid = wandb.Image(outs, caption='Output Images')
    mse_grid = wandb.Image(torch.abs(ins-outs)**2, caption='MSE')
    pse_grid = wandb.Image(PSELoss.get_loss_image(ins,outs), caption='PSE')
    
    wandb.log({'valid/images/inputs': inputs_grid,
               'valid/images/outputs': outputs_grid,
               'valid/images/PSE': pse_grid,
               'valid/images/MSE': mse_grid})

    # log_image_table(inputs[:8], outputs[:8], torch.abs(inputs[:8] - outputs[:8]), prefix = "valid/")

    # writer.add_scalar('val_loss', running_loss / ct, epoch)
    print("MSE", running_mse / ct, "PSE", running_pse / ct)
    if criterion == MSELoss:
        return running_mse / ct
    elif criterion == PSELoss:
        return running_pse / ct

# Train the model
num_epochs = 1000
best_loss = float('inf')
x = torch.randn(1, 3, 96, 96).to(device).requires_grad_(True)
make_dot(model(x), params=dict(model.named_parameters())).render("model_architecture", format="png")
make_dot(model(x), params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("model_architecture_verbose", format="png")
for epoch in range(num_epochs):
    print("Epoch:", epoch)
    print("Train")
    train()
    print("Validation")
    val_loss = validate()
    if val_loss < best_loss:
        print("SAVING THE MODEL")
        best_loss = val_loss
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   "models/"+model_name+".pt")
    
    
wandb.finish()
# writer.close()