from torch import nn
import torch

class MWLoss(nn.Module):
    def __init__(self, k, b):
        super().__init__()
        self.k = k
        self.b = b
        self.MSE = nn.MSELoss(reduction='none')

    def forward(self, pred, gt):
        mw = self.k * gt + self.b * torch.where(gt > 0.5, 1, 0)
        loss = torch.mean(mw * self.MSE(pred, gt))
        return loss