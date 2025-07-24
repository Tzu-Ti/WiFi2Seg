from typing import List
import torch
from torch import nn
import einops

import sys
sys.path.append('..')
from MHMA import MultiHeadMixAttention
from utils import Activation


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout=0.1):
        """
        Args:
            d_model (int): Dimension of the model.
            n_head (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.MA = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def _sa_block(self, x, src_mask=None):
        x = self.MA(x, x, x, attn_mask=src_mask, need_weights=False)[0]
        x = self.dropout(x)
        return x

    def forward(self, x, src_mask=None):
        new_x = self.norm(x)
        x = x + self._sa_block(new_x, src_mask=src_mask)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int,
                 dropout=0.1):
        super().__init__()

        self.l_norm = nn.LayerNorm(d_model)
        self.MA = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def _ca_block(self, q, k, v, src_mask=None):
        x = self.MA(q, k, v, attn_mask=src_mask, need_weights=False)[0]
        x = self.dropout(x)
        return x

    def forward(self, q, k, v, src_mask=None):
        new_q = self.l_norm(q)
        k = self.l_norm(k)
        v = self.l_norm(v)
        q = q + self._ca_block(new_q, k, v, src_mask=src_mask)
        q = self.dropout(q)
        return q
    
class MixAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int,
                 dropout=0.1):
        """
        Args:
            d_model (int): Dimension of the model.
            n_head (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.a_norm = nn.LayerNorm(d_model)
        self.b_norm = nn.LayerNorm(d_model)
        self.MA = MultiHeadMixAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def _ma_block(self, a, b, src_mask=None):
        a, b = self.MA(a, b, attn_mask=src_mask)
        return self.dropout(a), self.dropout(b)

    def forward(self, a, b, src_mask=None):
        new_a = self.a_norm(a)
        new_b = self.b_norm(b)
        new_a, new_b = self._ma_block(new_a, new_b, src_mask=src_mask)
        a = a + new_a
        b = b + new_b
        return a, b

class ReverseAttentionBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.amp_sigmoid = nn.Sigmoid()
        self.pha_sigmoid = nn.Sigmoid()
        self.amp_ff = FFBlock(d_model=d_model)
        self.pha_ff = FFBlock(d_model=d_model)

    def forward(self, amp, pha, residual_amp, residual_pha):
        amp_rev = residual_amp * self.amp_sigmoid(-amp)
        pha_rev = residual_pha * self.pha_sigmoid(-pha)
        amp = self.amp_ff(amp_rev)
        pha = self.pha_ff(pha_rev)

        return amp, pha, amp_rev, pha_rev
    
class AdaptiveCNN(nn.Module):
    def __init__(self, d_model: int, kernel_sizes: List[int], dropout=0.1,
                 activation='relu'):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        self.conv_blocks = self._make_layers(d_model, kernel_sizes, activation=activation, dropout=dropout)
        self.weight_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            Activation(activation),
            nn.Linear(d_model, 1)
        )
        self.softmax = nn.Softmax(dim=-1) 

    def _make_layers(self, d_model: int, kernel_sizes: int, activation: str, dropout: float):
        layers = []
        for kernel_size in kernel_sizes:
            layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(d_model),
                nn.Dropout(dropout),
                Activation(activation)
            ))
        return nn.ModuleList(layers)

    def _msc_forward(self, x):
        multiscale_features = None
        for conv in self.conv_blocks:
            if multiscale_features is None:
                multiscale_features = conv(x).unsqueeze(-1)  # Add a dimension for concatenation
            else:
                multiscale_features = torch.cat((multiscale_features, conv(x).unsqueeze(-1)), dim=-1) # Concatenate along the last dimension
        multiscale_features = einops.rearrange(multiscale_features, 'b c n k -> b n c k')
        f = self.weight_ff(einops.rearrange(multiscale_features, 'b n c k -> b n k c'))
        weights = self.softmax(f)
        out = torch.matmul(multiscale_features, weights)
        out = out.squeeze(-1)  # Remove the last dimension after concatenation
        return out


    def forward(self, x):
        new_x = self.norm(x)
        new_x = einops.rearrange(new_x, 'b n c -> b c n')  # Rearrange to (batch, channels, length)
        new_x = self._msc_forward(new_x)
        x = x + new_x
        return x


class TemperalStream(nn.Module):
    def __init__(self, d_model: int, n_head: int, kernel_sizes: List[int],
                  dropout=0.1,):
        super().__init__()
        
        self.attn = MixAttentionBlock(d_model=d_model, n_head=n_head, dropout=dropout)

        self.amp_msc = AdaptiveCNN(d_model=d_model, kernel_sizes=kernel_sizes, dropout=dropout)
        self.pha_msc = AdaptiveCNN(d_model=d_model, kernel_sizes=kernel_sizes, dropout=dropout)


    def forward(self, amp, pha):
        amp, pha = self.attn(amp, pha)
        amp_msc = self.amp_msc(amp)
        pha_msc = self.pha_msc(pha)

        return amp_msc, pha_msc
    
class ChannelStream(TemperalStream):
    def __init__(self, d_model: int, n_head: int, kernel_sizes: List[int],
                 dropout=0.1,
                 use_reverse=False):
        super().__init__(d_model=d_model, n_head=n_head, kernel_sizes=kernel_sizes, dropout=dropout)

        self.use_reverse = use_reverse

        if self.use_reverse:
            self.reverse_attention = ReverseAttentionBlock(d_model=d_model)

    def forward(self, amp, pha):
        amp, pha = self.attn(amp, pha)
        amp_msc = self.amp_msc(amp)
        pha_msc = self.pha_msc(pha)

        if self.use_reverse:
            amp, pha, amp_rev, pha_rev = self.reverse_attention(amp_msc, pha_msc, amp, pha)
            return amp, pha, amp_rev, pha_rev

        return amp_msc, pha_msc

class FFBlock(nn.Module):
    def __init__(self, d_model: int,
                 activation='relu', dropout=0.1):
        super().__init__()

        self.l_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            Activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        new_x = self.l_norm(x)
        new_x = self.ff(new_x)
        x = x + new_x
        return x

class CrossAggregationBlock(nn.Module):
    def __init__(self, length: int, RxTx_num: int,
                 d_model: int, latent_dim: int,
                 n_head: int,
                 activation='relu', dropout=0.1):
        super().__init__()

        d_model_temporal = d_model * RxTx_num

        self.in_proj = nn.Sequential(
            nn.Linear(d_model_temporal*2, latent_dim),
            nn.LayerNorm(latent_dim),
            Activation(activation),
            nn.Dropout(dropout)
        )
        self.sa_block = SelfAttentionBlock(d_model=latent_dim, n_head=n_head)
        self.ff_in_block = FFBlock(d_model=latent_dim)
        self.ca_block = CrossAttentionBlock(d_model=latent_dim, n_head=n_head)
        self.ff_block = FFBlock(d_model=latent_dim)

        self.learnable_query = nn.Parameter(torch.empty(length, latent_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.learnable_query, gain=1.0)

    def forward(self, amp, pha):
        csi = torch.cat([amp, pha], dim=2)
        csi = self.in_proj(csi)
        csi = self.sa_block(csi)
        csi = self.ff_in_block(csi)

        b = csi.size(0)
        Q = einops.repeat(self.learnable_query, 'l d -> b l d', b=b)
        Q = self.ca_block(Q, csi, csi)
        Q = self.ff_block(Q)
        Q = einops.rearrange(Q, 'b l d -> b (l d)')

        return Q

if __name__ == '__main__':
    # Example usage of SelfAttentionBlock
    # import torch

    # sa_block = SelfAttentionBlock(d_model=1536, n_head=6)
    # x = torch.randn(32, 51, 1536)  # batch_size=32, seq_len=51, d_model=1536
    # output = sa_block(x)
    # print(output.shape)  # Should be [32, 51, 1536]

    # Example usage of MixAttentionBlock
    # import torch
    # ma_block = MixAttentionBlock(d_model=1536, n_head=6)
    # a = torch.randn(32, 51, 1536)  # batch_size=32, seq_len=51, d_model=1536
    # b = torch.randn(32, 51, 1536)  # batch_size=32, seq_len=51, d_model=1536
    # output_a, output_b = ma_block(a, b)
    # print(output_a.shape)  # Should be [32, 51, 1536]
    # print(output_b.shape)  # Should be [32, 51, 1536]

    # Example usage of AdaptiveCNN
    # import torch
    # adaptive_cnn = AdaptiveCNN(d_model=1536, kernel_sizes=[3, 5, 7], dropout=0.1, activation='relu')
    # x = torch.randn(32, 51, 1536)  # batch_size=32, d_model=1536
    # x = adaptive_cnn(x)
    # print(x.shape)  # Should be [32, 51, 1536] after processing through the AdaptiveCNN

    # Example usage of TemperalStream or ChannelStream
    # import torch
    # cs = ChannelStream(d_model=306, n_head=6, kernel_sizes=[3, 5, 7], dropout=0.1)
    # amp = torch.randn(32, 256, 306)
    # pha = torch.randn(32, 256, 306) 
    # amp_msc, pha_msc = cs(amp, pha)

    # Example usage of CrossAggregationBlock
    import torch
    CA = CrossAggregationBlock(
        length=1, RxTx_num=6,
        d_model=256, latent_dim=512, n_head=8)
    amp = torch.randn(4, 51, 1536)
    pha = torch.randn(4, 51, 1536)
    Q = CA(amp, pha)