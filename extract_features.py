import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, TensorDataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

from models import AutoEncoder, AutoEncoder_VGG, AutoEncoder_ResNet

def compute_features(model, data_train, data_test, k, n_splits, device, compute_test = True):
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.05, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
    ])

    dataset_full = ImageFolder(data_train, transform=None)
    if compute_test:
        data_test = ImageFolder(data_test, transform=val_transform)

    # choose fold to train on
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=12345)
    all_splits = [k for k in kf.split(dataset_full)]

    train_indexes, val_indexes = all_splits[k]
    train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

    train_data = ImageFolder(data_train, transform=train_transform)
    val_data = ImageFolder(data_train, transform=val_transform)

    data_train = Subset(train_data, train_indexes)
    data_val = Subset(val_data, val_indexes)

    print("Computing Features Train...")
    features_train = []
    for image, _ in tqdm(data_train):
        image = image.to(device)
        with torch.no_grad():
            features_train.append(model(torch.stack([image]))[0])
        torch.cuda.empty_cache()  # Free GPU memory
    features_train = torch.cat(features_train)
    targets_train = torch.tensor([data_train.dataset.targets[idx] for idx in range(len(data_train))])
    dataset_train = TensorDataset(features_train.cpu(), targets_train.cpu())

    print("Computing Features Valid...")
    features_valid = []
    for image, _ in tqdm(data_val):
        image = image.to(device)
        with torch.no_grad():
            features_valid.append(model(torch.stack([image]))[0])
        torch.cuda.empty_cache()  # Free GPU memory
    features_valid = torch.cat(features_valid)
    targets_valid = torch.tensor([data_val.dataset.targets[idx] for idx in range(len(data_val))])
    dataset_valid = TensorDataset(features_valid.cpu(), targets_valid.cpu())
    
    if compute_test:
        print("Computing Features Test...")
        features_test = []
        for image, _ in tqdm(data_test):
            image = image.to(device)
            with torch.no_grad():
                features_test.append(model(torch.stack([image]))[0])
            torch.cuda.empty_cache()  # Free GPU memory
        features_test = torch.cat(features_test)
        targets_test = torch.tensor([target for _, target in data_test.imgs])
        dataset_test = TensorDataset(features_test.cpu(), targets_test.cpu())
    else:
        dataset_test = None
        
    return dataset_train, dataset_valid, dataset_test



# Set your parameters here
data_train = "train/"
data_test = "test/"
k = 0  # Fold number
n_splits = 5  # Number of splits for cross-validation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder_model_type = "CNN"
autoencoder_model_name = "AUTOENCODER_PRETRAIN_CNN_MSE"


# Set the directory to save the datasets
save_dir = "features/"

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

if autoencoder_model_type == "CNN":
    features_dim=256
    autoencoder_model = AutoEncoder(latent_dim=256)
elif autoencoder_model_type == "VGG":
    features_dim=256
    autoencoder_model = AutoEncoder_VGG(latent_dim=256)
elif autoencoder_model_type == "RESNET":
    features_dim=1024
    autoencoder_model = AutoEncoder_ResNet(latent_dim=1024)
elif autoencoder_model_type == "MASKED":
    features_dim=1024
    autoencoder_model = mae_vit_large_patch16(img_size=96)
if os.path.exists("models/"+autoencoder_model_name+".pt"):
    # load the model
    print("Loading Model...")
    checkpoint = torch.load("models/"+autoencoder_model_name+".pt")
    autoencoder_model.load_state_dict(checkpoint['model_state_dict'])
            
autoencoder_model.to(device)
compute_test = True
# Compute features and save datasets for each fold
for k in range(n_splits):
    dataset_train, dataset_valid, dataset_test = compute_features(autoencoder_model, data_train, data_test, k, n_splits, device, compute_test = compute_test)
    
    # Set the file paths to save the datasets
    dataset_train_path = os.path.join(save_dir, f"dataset_train_fold{k}.pt")
    dataset_valid_path = os.path.join(save_dir, f"dataset_valid_fold{k}.pt")
    dataset_test_path = os.path.join(save_dir, f"dataset_test_fold{k}.pt")

    # Save the datasets
    torch.save(dataset_train, dataset_train_path)
    torch.save(dataset_valid, dataset_valid_path)
    if compute_test:
        torch.save(dataset_test, dataset_test_path)

    print(f"Fold {k}: Datasets saved successfully!")
    
    compute_test = False