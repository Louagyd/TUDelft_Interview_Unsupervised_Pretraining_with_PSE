import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

import wandb

from models import AutoEncoder, AutoEncoder_VGG, AutoEncoder_ResNet
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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

# Add augmentation to the training DataLoader
train_loader.dataset.transform = transforms.Compose([
    augmentation,
    preprocess
])
    
# Define the model and optimizer
# model = AutoEncoder(latent_dim=256)
# model = AutoEncoder_VGG(latent_dim=256)
model = AutoEncoder_ResNet(latent_dim=256)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the loss function and device
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using Device", device)
model.to(device)

# Define the TensorBoard writer
# writer = SummaryWriter('logs')
wandb.watch(model, log='all')

def log_image_table(input_images, output_images, diffs, prefix = "train"):
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["input", "output", "ABS"])
    for img, pred, dif in zip(input_images.to("cpu"), 
                                   output_images.to("cpu"),
                                   diffs.to("cpu")):
        table.add_data(wandb.Image(img.detach().numpy().transpose((1,2,0))*255), 
                       wandb.Image(pred.detach().numpy().transpose((1,2,0))*255),
                       wandb.Image(dif.detach().numpy().transpose((1,2,0))*255))
    wandb.log({prefix+"images":table}, commit=False)

log_interval = 100
# Define the training loop
def train():
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_grad_norm = 0.0
    
    for i, (inputs, _) in enumerate(train_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        latents, outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        running_loss += loss.item()
        running_grad_norm += grad_norm
        
        if i % log_interval == log_interval - 1:
            plt.imsave("input_train.jpg", inputs[0,...].detach().cpu().numpy().transpose((1,2,0)))
            plt.imsave("output_train.jpg", outputs[0,...].detach().cpu().numpy().transpose((1,2,0)))
            print("Iteration", i,
                  "LOSS", running_loss / log_interval,
                  "GN", running_grad_norm / log_interval)
            wandb.log({"train/Loss": running_loss / log_interval, 
                       "train/Gradient_Norm": running_grad_norm / log_interval})
            # writer.add_scalar('train_loss', running_loss / log_interval, i)
            # writer.add_scalar('grad_norm', running_grad_norm / log_interval, i)
            running_loss = 0.0
            running_acc = 0.0
            running_grad_norm = 0.0
        if i == 1:
          inputs_grid = wandb.Image(inputs[:16] , caption='Input Images')
          outputs_grid = wandb.Image(outputs[:16], caption='Output Images')
          diff_grid = wandb.Image(torch.abs(inputs[:16]-outputs[:16]), caption='MSE Images')
          wandb.log({'train/images/inputs': inputs_grid,
                     'train/images/outputs': outputs_grid,
                     'train/images/diffs': diff_grid})


            # log_image_table(inputs[:8], outputs_grid[:8], torch.abs(inputs[:8] - outputs[:8]), prefix = "train/")

            
# Define the validation loop
def validate():
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    ct = 0
    with torch.no_grad():
        for inputs, _ in valid_loader:
            inputs = inputs.to(device)
            latents, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_loss += loss.item()
            ct += 1
    
    plt.imsave("input_valid.jpg", inputs[0,...].detach().cpu().numpy().transpose((1,2,0)))
    plt.imsave("output_valid.jpg", outputs[0,...].detach().cpu().numpy().transpose((1,2,0)))
    wandb.log({"valid/Loss": running_loss / ct})
    
    rnd_ind = np.random.randint(8,58)
    ins = torch.concat([inputs[:8], inputs[rnd_ind:rnd_ind+8]])
    outs = torch.concat([outputs[:8], outputs[rnd_ind:rnd_ind+8]])
    inputs_grid = wandb.Image(ins , caption='Input Images')
    outputs_grid = wandb.Image(outs, caption='Output Images')
    diff_grid = wandb.Image(torch.abs(ins-outs), caption='MSE Images')
    wandb.log({'valid/images/inputs': inputs_grid,
               'valid/images/outputs': outputs_grid,
               'valid/images/diffs': diff_grid})

    # log_image_table(inputs[:8], outputs[:8], torch.abs(inputs[:8] - outputs[:8]), prefix = "valid/")

    # writer.add_scalar('val_loss', running_loss / ct, epoch)
    print("LOSS", running_loss / ct)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    print("Starting Epoch:", epoch)
    print("Train")
    train()
    print("Validation")
    validate()

wandb.finish()
# writer.close()