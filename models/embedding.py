import torch
from torch import nn
import einops

import sys
sys.path.append('models')
from utils import Activation

class ProjectionBlock(nn.Module):
    def __init__(self, length: int, RxTx_num: int,
                 in_channels: int, d_model: int,
                 activation='relu', dropout=0.1):
        super().__init__()

        self.length = length
        self.RxTx_num = RxTx_num
        self.in_channels = in_channels

        self.channel_projection = nn.Sequential(
            nn.AvgPool1d(4, 2, 1),
            nn.Linear(in_channels//2, d_model),
            nn.LayerNorm(d_model),
            Activation(activation)
        )

        # length == 51
        self.pool_3 = nn.AvgPool1d(3, 1, 1)
        self.pool_5 = nn.AvgPool1d(5, 1, 2)
        self.temporal_projection = nn.Sequential(
            nn.Linear(length, length),
            nn.LayerNorm(length),
            Activation(activation)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = einops.rearrange(x, 'b l n c -> b (l n) c')
        x = self.channel_projection(x)
        x = einops.rearrange(x, 'b (l n) c -> b l n c', n=self.RxTx_num)

        x = einops.rearrange(x, 'b l n c -> b c n l')
        x = einops.rearrange(x, 'b c n l -> b (c n) l')
        x = (x*4 + self.pool_3(x)*2 + self.pool_5(x)*1)/7
        x = self.temporal_projection(x)
        x = einops.rearrange(x, 'b (c n) l -> b c n l', n=self.RxTx_num)
        x = einops.rearrange(x, 'b c n l -> b l n c')
        x = einops.rearrange(x, 'b l n c -> b l (n c)')
        
        x = self.dropout(x)

        return x

class GaussianRangeEmbedding(nn.Module):
    def __init__(self, length: int, RxTx_num: int,
                 in_channels: int, d_model: int,
                 k=12):
        super().__init__()

        out_channels = d_model * RxTx_num
        self.projection = ProjectionBlock(length=length, RxTx_num=RxTx_num, in_channels=in_channels, d_model=d_model)

        self.embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(k, out_channels), requires_grad=True),
            gain=1.0
        )

        self.mu = nn.Parameter(torch.linspace(0, length, k, requires_grad=True))
        self.std = nn.Parameter(torch.ones(k, requires_grad=True) * length / 3)
        self.positions = torch.arange(0, length, 1).float().unsqueeze(1).repeat(1, k)
        self.positions = nn.Parameter(self.positions, requires_grad=False)
        self.const = -0.5 * torch.log(torch.tensor(2 * torch.pi))
        self.const = nn.Parameter(self.const, requires_grad=False)

    def _ln_pdf(self, pos, mu, std):
        a = (pos - mu) / std
        ln_p = -torch.log(std) + self.const - 0.5 * a**2
        return nn.functional.softmax(ln_p, dim=1)

    def forward(self, x):
        x = self.projection(x)

        pdf = self._ln_pdf(self.positions, self.mu, self.std)
        embedding = torch.matmul(pdf, self.embedding)

        x = x + embedding
        return x


if __name__ == '__main__':
    GRE = GaussianRangeEmbedding(length=51, RxTx_num=6,
                                 in_channels=2025, d_model=256)
    input = torch.randn([1, 51, 6, 2025])
    output = GRE(input)
    print(output.shape)  # Expected shape: [1, 51, 1536]
