from typing import List
from torch import nn
import einops
import copy

import sys
sys.path.append('models')
from modules import SelfAttentionBlock, TemperalStream, ChannelStream
from utils import random_src_mask

class HybridAttentionLayer(nn.Module):
    def __init__(self, length: int, RxTx_num: int,
                 d_model: int, n_head: int,
                 kernel_sizes: List[int],
                 mask=None,
                 use_reverse=False):
        super().__init__()
        """
        Args:
            length (int): Length of the input sequence.
            RxTx_num (int): Number of RxTx pairs.
            d_model (int): Dimension of the model.
            n_head (int): Number of attention heads.
            mask (float, optional): Ratio for random source masking. Default is None.
        """
        # experiment zone
        self.use_reverse = use_reverse

        ###

        self.length = length
        self.RxTx_num = RxTx_num
        self.mask = mask

        d_model_temporal = d_model * RxTx_num
        d_model_channel = length * RxTx_num
        self.amp_sa = SelfAttentionBlock(d_model=d_model_temporal, n_head=n_head)
        self.pha_sa = SelfAttentionBlock(d_model=d_model_temporal, n_head=n_head)

        self.temporal_stream = TemperalStream(d_model=d_model_temporal, n_head=n_head, kernel_sizes=kernel_sizes)
        self.channel_stream = ChannelStream(d_model=d_model_channel, n_head=n_head, kernel_sizes=kernel_sizes,
                                            use_reverse=use_reverse)

        self.amp_norm_temporal = nn.LayerNorm(d_model_temporal)
        self.pha_norm_temporal = nn.LayerNorm(d_model_temporal)
        self.amp_norm_channel = nn.LayerNorm(d_model_temporal)
        self.pha_norm_channel = nn.LayerNorm(d_model_temporal)

    def forward(self, amp, pha):
        """
        Args:
            amp (Tensor): Amplitude data, Shape: [batch, frame_num(51), d_model_tem(1536)]
            pha (Tensor): Phase data, Shape: [batch, frame_num(51), d_model_tem(1536)]

        """
        if self.mask:
            src_mask = random_src_mask(amp.shape, ratio=self.mask)
        else:
            src_mask = None
        amp = self.amp_sa(amp, src_mask=src_mask)
        pha = self.pha_sa(pha, src_mask=src_mask)

        # temporal stream
        amp, pha = self.temporal_stream(amp, pha)

        # channel stream
        amp_cs = einops.rearrange(amp, 'b l (n c) -> b c (n l)', n=self.RxTx_num)
        pha_cs = einops.rearrange(pha, 'b l (n c) -> b c (n l)', n=self.RxTx_num)
        if self.use_reverse:
            amp_cs, pha_cs, amp_rev, pha_rev = self.channel_stream(amp_cs, pha_cs)
        else:
            amp_cs, pha_cs = self.channel_stream(amp_cs, pha_cs)
        amp_cs = einops.rearrange(amp_cs, 'b c (n l) -> b l (n c)', n=self.RxTx_num)
        pha_cs = einops.rearrange(pha_cs, 'b c (n l) -> b l (n c)', n=self.RxTx_num)

        # norm & add
        amp = self.amp_norm_temporal(amp) + self.amp_norm_channel(amp_cs)
        pha = self.pha_norm_temporal(pha) + self.pha_norm_channel(pha_cs)

        if self.use_reverse:
            return amp_cs, pha_cs, amp_rev, pha_rev
        else:
            return amp, pha
    
class HybridAttention(nn.Module):
    def __init__(self, layer, num_layers: int):
        super().__init__()
        
        self.use_reverse = layer.use_reverse
        self.layers = self._make_layer(layer, num_layers=num_layers)
        
    def _make_layer(self, layer, num_layers: int):
        return nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, amp, pha):
        if self.use_reverse:
            for layer in self.layers:
                amp, pha, amp_rev, pha_rev = layer(amp, pha)
            return amp, pha, amp_rev, pha_rev
        else:
            for layer in self.layers:
                amp, pha = layer(amp, pha)
            return amp, pha

if __name__ == '__main__':
    import torch

    length = 51
    RxTx_num = 6
    d_model = 256
    n_head = 6
    kernel_sizes = [3, 5, 7]
    num_layers = 3
    mask = None  # Set to None for no masking

    amp = torch.randn(10, 51, 1536)  # batch_size=10
    pha = torch.randn(10, 51, 1536)  # batch_size=10

    hybrid_layer = HybridAttentionLayer(length=length, RxTx_num=RxTx_num,
                                        d_model=d_model, n_head=n_head,
                                        kernel_sizes=kernel_sizes,
                                        mask=mask)
    hybrid_attention = HybridAttention(layer=hybrid_layer, num_layers=num_layers)
    amp, pha = hybrid_attention.forward(amp, pha)
    print("Amplitude output shape:", amp.shape)  # Should be [10, 51, 1536]
    print("Phase output shape:", pha.shape)  # Should be [10, 51, 1536]
