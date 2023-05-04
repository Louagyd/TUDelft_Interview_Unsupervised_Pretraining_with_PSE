import os

import torch
from torch.utils.data import DataLoader
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
            torch.nn.Dropout(p=0.1)
        )
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
        if self.just_init:
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)
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

    def on_test_epoch_end(self):
        loss_avg = torch.stack(self.test_losses).mean()
        acc_avg = torch.stack(self.test_accs).mean()
        print("Test Loss: ", loss_avg) 
        print("Test Accuracy: ", acc_avg) 
        self.test_losses.clear()  # free memory
        self.test_accs.clear()  # free memory

from sklearn.model_selection import KFold


class KFoldDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data: str = "train/",
            test_data: str = "test/",
            k: int = 1,  # fold number
            split_seed: int = 12345,  # split needs to be always the same for correct cross validation
            num_splits: int = 10,
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
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
            transforms.RandomRotation(degrees=30),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.ToTensor(),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
        ])
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        
    @property
    def num_classes() -> int:
        return 10

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

            self.data_train = ImageFolder(self.hparams.train_data, transform=self.train_transform)
            self.data_val = ImageFolder(self.hparams.train_data, transform=self.val_transform)
            self.data_train.samples = [self.data_train.samples[i] for i in train_indexes]
            self.data_val.samples = [self.data_val.samples[i] for i in val_indexes]
            
    def train_dataloader(self):
        out_loader = DataLoader(dataset=self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory, shuffle=True)
        return out_loader
        

    def val_dataloader(self):
        out_loader = DataLoader(dataset=self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)
        
        return out_loader
    
    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)


def evaluate(model, test_loader):
    correct = 0
    top2_correct = 0
    top4_correct = 0
    total = 0
    
    all_preds = []
    # loop over the test data and calculate the accuracy
    for images, labels in tqdm(test_loader):
        print(1)
        outputs = model(images)
        print(2)
        preds = F.softmax(outputs, dim=1)
        print(3)
        all_preds.append(preds)
        print(4)
        _, predicted = torch.max(outputs.data, 1)
        print(5)
        # _, top2_predictions = torch.topk(outputs.data, 2)
        # _, top4_predictions = torch.topk(outputs.data, 4)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # top2_correct += (top2_predictions[:, 0] == labels.unsqueeze(1)).sum().item()
        # top4_correct += (top4_predictions[:, 0] == labels.unsqueeze(1)).sum().item()
    accuracy = 100 * correct / total
    # accuracy_top2 = 100 * top2_correct / total
    # accuracy_top4 = 100 * top4_correct / total
    return torch.cat(all_preds, dim=0), accuracy

features_dim=256
autoencoder_model = AutoEncoder(latent_dim=256)
# autoencoder_model = AutoEncoder_VGG(latent_dim=256)
# autoencoder_model = AutoEncoder_ResNet(latent_dim=1024)
# autoencoder_model = mae_vit_large_patch16(img_size=96)


autoencoder_model_name = 'AUTOENCODER_PRETRAIN_CNN_MSE'
if os.path.exists("models/"+autoencoder_model_name+".pt"):
    # load the model
    checkpoint = torch.load("models/"+autoencoder_model_name+".pt")
    autoencoder_model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using Device", device)

# from pytorch_lightning.loggers import WandbLogger
# wandb_logger = WandbLogger(project='tudelft_interview')
early_stop_callback = pl.callbacks.EarlyStopping(monitor="valid_loss" , patience=20)
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="models/"+autoencoder_model_name, save_top_k=1, monitor="valid_loss")

model = DownstreamWrapper(autoencoder_model, num_classes=10, features_dim=features_dim)
model.to(device)
num_splits = 5
all_preds = []
for k in range(num_splits):
    data_module = KFoldDataModule(k = k, num_splits = num_splits)
    data_module.setup()
    
    trainer = pl.Trainer(max_epochs=500, gradient_clip_val=0.5, callbacks=[early_stop_callback,
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
