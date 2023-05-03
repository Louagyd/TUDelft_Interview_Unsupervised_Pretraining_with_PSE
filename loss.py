import torch.nn.functional as F
import torch.nn as nn
import torch

class PSELoss(nn.Module):
    def __init__(self, sigma):
        super(PSELoss, self).__init__()
        self.sigma = sigma

    def forward(self, y_true, y_pred):
        n = y_true.size(0)

        diff = torch.abs(y_true - y_pred)
        kernel = self.get_kernel(diff.size(), device=diff.device)
        conv = F.conv2d(diff, kernel, padding='same')
        loss = torch.mean(conv**2)

        return loss
    
    def get_loss_image(self, y_true, y_pred):
        
        diff = torch.abs(y_true - y_pred)
        kernel = self.get_kernel(diff.size(), device=diff.device)
        conv = F.conv2d(diff, kernel, padding='same')
        loss_image = conv**2

        return loss_image
    
    def get_kernel(self, shape, device):
        ksize = int(2 * self.sigma + 1)
        kernel = torch.zeros(3, 3, ksize, ksize).to(device)
        for c in range(3):
            for x in range(-self.sigma, self.sigma + 1):
                for y in range(-self.sigma, self.sigma + 1):
                    value = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * self.sigma**2)))
                    kernel[c,c,x+self.sigma,y+self.sigma] = value
        
        kernel = kernel / kernel.sum()
        return kernel

