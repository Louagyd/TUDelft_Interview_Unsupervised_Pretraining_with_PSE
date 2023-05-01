import torch.nn.functional as F
import torch.nn as nn

class PSELoss(nn.Module):
    def __init__(self, sigma):
        super(PSELoss, self).__init__()
        self.sigma = sigma

    def forward(self, y_true, y_pred):
        n = y_true.size(0)
        y_true = y_true.view(n, -1)
        y_pred = y_pred.view(n, -1)

        diff = y_true - y_pred
        kernel = self.get_kernel(diff.size(), device=diff.device)
        loss = F.conv2d(diff.view(n, 1, *diff.shape[1:]), kernel, padding=1).view(n, -1).mean()
        return loss

    def get_kernel(self, shape, device):
        ksize = int(2 * self.sigma + 1)
        kernel = torch.zeros(1, 1, ksize, ksize).to(device)
        for x in range(-self.sigma, self.sigma + 1):
            for y in range(-self.sigma, self.sigma + 1):
                value = torch.exp(-(x**2 + y**2) / (2 * self.sigma**2))
                kernel[:,:,x+self.sigma,y+self.sigma] = value
        kernel = kernel / kernel.sum()
        return kernel