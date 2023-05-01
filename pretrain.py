import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import AutoEncoder
from loss import PSELoss

# Define transforms for preprocessing and data augmentation
preprocess = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

# Add augmentation to the training DataLoader
train_loader.dataset.transform = transforms.Compose([
    augmentation,
    preprocess
])
    
# Define the model and optimizer
model = AutoEncoder(latent_dim=256)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the loss function and device
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the TensorBoard writer
writer = SummaryWriter('logs')

# Define a function to calculate accuracy
def accuracy(outputs, targets):
    preds = torch.round(torch.sigmoid(outputs))
    correct = (preds == targets).sum().float()
    accuracy = correct / len(targets)
    return accuracy

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
        acc = accuracy(outputs, inputs)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_acc += acc.item() * inputs.size(0)
        running_grad_norm += grad_norm
        
        if i % 100 == 99:
            batch_size = len(train_loader)
            writer.add_scalar('train_loss', running_loss / batch_size, i)
            writer.add_scalar('train_acc', running_acc / batch_size, i)
            writer.add_scalar('grad_norm', running_grad_norm / batch_size, i)
            running_loss = 0.0
            running_acc = 0.0
            running_grad_norm = 0.0

# Define the validation loop
def validate():
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            latents, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            acc = accuracy(outputs, inputs)
            running_loss += loss.item() * inputs.size(0)
            running_acc += acc.item() * inputs.size(0)
    
    batch_size = len(val_loader.dataset)
    writer.add_scalar('val_loss', running_loss / batch_size, epoch)
    writer.add_scalar('val_acc', running_acc / batch_size, epoch)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    train()
    validate()

writer.close()