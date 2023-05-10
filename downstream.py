import torch
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pytorch_lightning as pl

from models import AutoEncoder, AutoEncoder_VGG, AutoEncoder_ResNet
from models_mae import mae_vit_large_patch16

# import wandb
# wandb.login()

from torchmetrics import Accuracy

import os
from tqdm import tqdm
import torch.nn.functional as F

from pytorch_lightning.callbacks import TQDMProgressBar


class DownstreamWrapper(pl.LightningModule):
    def __init__(self, autoencoder_model, num_classes, features_dim, freeze_encoder=False, use_features = 0, num_layers = 2):
        super().__init__()
        if use_features == 0:
            self.autoencoder_model = autoencoder_model
            
        self.features_dim = features_dim
        self.freeze_encoder = freeze_encoder
        self.use_features = use_features
        
        layers = [
            torch.nn.Linear(features_dim, 128),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(p=0.1)
        ]
        for _ in range(num_layers - 1):
            layers = layers + [
                torch.nn.Linear(128, 128),
                torch.nn.Sigmoid(),
                torch.nn.Dropout()
            ]
            
        self.hidden = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(128, num_classes)
        
        self.training_losses = []
        self.training_accs = []
        self.validation_losses = []
        self.validation_accs = []
        self.test_losses = []
        self.test_accs = []
        self.test_preds = []
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        self.accuracy = Accuracy('multiclass', num_classes = num_classes)
        
        self.just_init = True
        
    def forward(self, x):
        if self.use_features == 0:
            with torch.no_grad():
                features, _ = self.autoencoder_model(x)

            if self.freeze_encoder:
                features = features.detach()

            hidden = self.hidden(features)
        else:
            hidden = self.hidden(x)
            
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
        if self.just_init and self.use_features == 0:
            import numpy as np
            import matplotlib.pyplot as plt
            plt.imsave("train_samples.jpg", np.concatenate([x[ii].cpu().numpy().transpose((1,2,0)) for ii in range(10)]))
            self.just_init = False
            
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
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.test_losses.append(loss)
        self.test_accs.append(acc)
        self.test_preds.append(F.softmax(logits, dim=1))
    
    def configure_optimizers(self):
        if self.freeze_encoder:
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-4)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=40)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_loss'}
    
#         if self.freeze_encoder:
#             optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-4)
#         else:
#             optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)

#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=100)
#         return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_loss'}
        
    def on_train_epoch_end(self):
        loss_avg = torch.stack(self.training_losses).mean()
        acc_avg = torch.stack(self.training_accs).mean()
        self.log("train_loss", loss_avg, prog_bar=True)
        self.log("train_acc", acc_avg, prog_bar=True)
        self.training_losses.clear()  # free memory
        self.training_accs.clear()  # free memory
        torch.cuda.empty_cache()  # free GPU memory

    def on_validation_epoch_end(self):
        loss_avg = torch.stack(self.validation_losses).mean()
        acc_avg = torch.stack(self.validation_accs).mean()
        self.log("valid_loss", loss_avg, prog_bar=True)
        self.log("valid_acc", acc_avg, prog_bar=True)
        self.validation_losses.clear()  # free memory
        self.validation_accs.clear()  # free memory
        torch.cuda.empty_cache()  # free GPU memory

    def on_test_epoch_end(self):
        loss_avg = torch.stack(self.test_losses).mean()
        acc_avg = torch.stack(self.test_accs).mean()
        print("Test Loss: ", loss_avg) 
        print("Test Accuracy: ", acc_avg) 
        self.test_losses.clear()  # free memory
        self.test_accs.clear()  # free memory
        torch.cuda.empty_cache()  # free GPU memory
        


from sklearn.model_selection import KFold


class KFoldDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data: str = "train/",
            test_data: str = "test/",
            k: int = 1,  # fold number
            split_seed: int = 12345,  # split needs to be always the same for correct cross validation
            num_splits: int = 10,
            numclasses: int = 10,
            batch_size: int = 128,
            num_workers: int = 4,
            pin_memory: bool = False
        ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert 0 <= k < num_splits, "incorrect fold number"
        
        # Define transforms for data augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=96,scale=(0.8, 1.0),ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.05, hue=0.05),
            transforms.RandomRotation(degrees=10),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
            transforms.ToTensor(),
        ])
        
        # self.train_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(96),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
        #     transforms.RandomRotation(degrees=30),
        #     transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        #     transforms.ToTensor(),
        # ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
        ])
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        
    @property
    def num_classes() -> int:
        return self.hparams.numclasses

    def setup(self, stage=None):
        if not self.data_train and not self.data_val:
            # Define dataset
            dataset_full = ImageFolder(self.hparams.train_data, transform=None)
            self.data_test = ImageFolder(self.hparams.test_data, transform=self.val_transform)

            # choose fold to train on
            kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)
            all_splits = [k for k in kf.split(dataset_full)]

            train_indexes, val_indexes = all_splits[self.hparams.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            train_data = ImageFolder(self.hparams.train_data, transform=self.train_transform)
            val_data = ImageFolder(self.hparams.train_data, transform=self.val_transform)

            self.data_train = Subset(train_data, train_indexes)
            self.data_val = Subset(val_data, val_indexes)
                
                # print(self.data_val.samples[0])
            
    def train_dataloader(self):
        out_loader = DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True
            )
        return out_loader
        

    def val_dataloader(self):
        out_loader = DataLoader(
                dataset=self.data_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory
            )
        return out_loader
    
    def test_dataloader(self):
        out_loader = DataLoader(
                dataset=self.data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory
            )
        return out_loader
    
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
    targets_train = torch.tensor([data_train.dataset.targets[idx] for idx in data_train.indices])
    dataset_train = TensorDataset(features_train.cpu(), targets_train.cpu())

    print("Computing Features Valid...")
    features_valid = []
    for image, _ in tqdm(data_val):
        image = image.to(device)
        with torch.no_grad():
            features_valid.append(model(torch.stack([image]))[0])
        torch.cuda.empty_cache()  # Free GPU memory
    features_valid = torch.cat(features_valid)
    targets_valid = torch.tensor([data_val.dataset.targets[idx] for idx in data_val.indices])
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

import torch.multiprocessing as mp
def main(args):
    
    num_epochs = args.num_epochs
    autoencoder_model_type = args.model_type
    autoencoder_model_name = args.model_name
    num_splits = args.num_splits
    batch_size = args.batch_size
    num_workers = args.num_workers
    freeze_encoder = args.freeze_encoder
    use_features = args.use_features
    num_layers = args.num_layers
    
    # mp.set_start_method('spawn', force=True)
    
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Using Device", device)

    all_preds = []
    compute_test = True
    for k in range(num_splits):
        if os.path.exists("models/"+autoencoder_model_name+".pt"):
            # load the model
            checkpoint = torch.load("models/"+autoencoder_model_name+".pt")
            autoencoder_model.load_state_dict(checkpoint['model_state_dict'])


        # from pytorch_lightning.loggers import WandbLogger
        # wandb_logger = WandbLogger(project='tudelft_interview')
        early_stop_callback = pl.callbacks.EarlyStopping(monitor="valid_loss" , patience=50)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="models/"+autoencoder_model_name, save_top_k=1, monitor="valid_loss")
        
        if use_features:
            if compute_test:
                dataset_train, dataset_valid, dataset_test = compute_features(autoencoder_model.to(device), 'train/', 'test/', k, num_splits, device, compute_test = compute_test)
            else:
                dataset_train, dataset_valid, _ = compute_features(autoencoder_model.to(device), 'train/', 'test/', k, num_splits, device, compute_test = compute_test)
                
            compute_test = False #only compute the test feature once
        
        encoder_wrapper = None if use_features else autoencoder_model
        model = DownstreamWrapper(encoder_wrapper, num_classes=10, features_dim=features_dim, freeze_encoder = freeze_encoder, use_features = use_features, num_layers = num_layers)
        model.to(device)

        data_module = KFoldDataModule(k = k, num_splits = num_splits, num_workers=num_workers)
        
        if use_features:
            data_module.data_train = dataset_train
            data_module.data_val = dataset_valid
            data_module.data_test = dataset_test
            
            
        data_module.setup()

        trainer = pl.Trainer(max_epochs=num_epochs, gradient_clip_val=0.5, callbacks=[early_stop_callback,
                                                                               TQDMProgressBar(refresh_rate=100),
                                                                               checkpoint_callback])
        trainer.fit(model, data_module)

        results = trainer.test(model, data_module, ckpt_path="best")
        # model_preds, accuracy = evaluate(model, test_loader)
        # print(f"Split {k}: accuracy = {accuracy*100:.4f}, accuracy_top2 = {accuracy_top2*100:.4f}, accuracy_top4 = {accuracy_top4*100:.4f}")
        all_preds.append(torch.cat(model.test_preds, dim=0))
        model.test_preds = []

    import numpy as np

    ensemble_preds = torch.mean(torch.stack(all_preds), dim=0)
    test_labels = np.asarray([data_module.data_test[i][1] for i in range(len(data_module.data_test))])
    ensemble_preds = ensemble_preds.cpu().numpy()

    ensemble_acc = (ensemble_preds.argmax(axis=1) == test_labels).mean()

    # Top-2 accuracy
    top2_preds = np.argsort(ensemble_preds, axis=1)[:, -2:]
    ensemble_top2_acc = np.mean(np.any(top2_preds == test_labels.reshape(-1, 1), axis=1))

    # Top-4 accuracy
    top4_preds = np.argsort(ensemble_preds, axis=1)[:, -4:]
    ensemble_top4_acc = np.mean(np.any(top4_preds == test_labels.reshape(-1, 1), axis=1))

    print(f"Ensemble accuracy: {ensemble_acc*100:.4f}")
    print(f"Ensemble top-2 accuracy: {ensemble_top2_acc*100:.4f}")
    print(f"Ensemble top-4 accuracy: {ensemble_top4_acc*100:.4f}")


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch Pretraining')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--model_type', type=str, default='CNN')
    parser.add_argument('--model_name', type=str, default='AUTOENCODER_PRETRAIN_CNN_MSE')
    parser.add_argument('--num_splits', type=float, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--freeze_encoder', type=int, default=0)
    parser.add_argument('--use_features', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    
    args = parser.parse_args()
    main(args)