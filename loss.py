import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.COSINE = nn.CosineEmbeddingLoss(margin=0.5)

    def forward(self, anchor, feature, label):
        amp1, pha1 = anchor
        amp2, pha2 = feature

        B = amp1.shape[0]
        amp1 = amp1.view(B, -1)
        pha1 = pha1.view(B, -1)
        amp2 = amp2.view(B, -1)
        pha2 = pha2.view(B, -1)

        loss = (self.COSINE(amp1, amp2, label) + self.COSINE(pha1, pha2, label)) / 2
        return loss
    
class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return kl_loss