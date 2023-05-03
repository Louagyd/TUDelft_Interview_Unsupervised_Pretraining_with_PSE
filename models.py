import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*9, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256*9)
        self.conv1 = nn.ConvTranspose2d(256, 256, 6, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(256, 128, 5, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = x.view(-1, 256, 3, 3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = nn.Sigmoid()(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return z, out

from torchvision.models import vgg19
class AutoEncoder_VGG(nn.Module):
    def __init__(self, latent_dim):
        super(AutoEncoder_VGG, self).__init__()
        
        # Define the encoder
        self.encoder = vgg19(pretrained=False).features
        
        # Replace the last maxpool layer with an adaptive pooling layer to handle variable input sizes
        self.encoder[-1] = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        
        # Define the projection layer
        self.projection_layer = nn.Linear(6 * 6 * 512, latent_dim)
        
        # Define the decoder
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projection_layer(x)
        
        # Decoder
        out = self.decoder(x)
        
        return x, out

    
import io


from torchvision.models import resnet18, ResNet18_Weights
class AutoEncoder_ResNet(nn.Module):
    def __init__(self, latent_dim):
        super(AutoEncoder_ResNet, self).__init__()
        
        # Define the encoder
        # self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = resnet18(pretrained = False)
        
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3)) # replace the last avgpool layer with adaptive pooling
        
        # Define the projection layer
        self.projection_layer = nn.Linear(512*3*3, latent_dim)
        
        # Define the decoder
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        # Encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = self.encoder.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.projection_layer(x)
        
        # Decoder
        out = self.decoder(x)
        
        return x, out

    