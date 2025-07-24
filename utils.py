import torch
from torch import nn, optim
import torchvision

def Activation(activation: str):
    activation = activation.lower()
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        return nn.ReLU()

def Optimizer(optimizer: str, **kwargs):
    optimizer = optimizer.lower()
    if optimizer == 'adam':
        return optim.Adam(kwargs['params'], lr=kwargs['lr'])
    elif optimizer == 'adamw':
        return optim.AdamW(kwargs['params'], lr=kwargs['lr'])
    
def random_src_mask(shape: list, ratio: float = 0.):
    mask = torch.rand(shape)
    mask = mask < ratio
    return mask.float()

def load_ckpt(model: nn.Module, ckpt_path: str, freeze: bool = False):
    """
    Load the model state dict from a lightning checkpoint file.
    
    Args:
        model (nn.Module): The model to load the state dict into.
        ckpt_path (str): The path to the checkpoint file.
        
    Returns:
        nn.Module: The model with loaded state dict.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = {k.replace("net.", ""): v for k, v in ckpt['state_dict'].items() if k.startswith("net.")}
    model.load_state_dict(state_dict)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
    return model

def make_img_grid(mask, out):
    images_to_log = torch.cat([
        mask[:6], out[:6],
        mask[6:12], out[6:12],
        mask[12:18], out[12:18]
    ], dim=0)
    img_grid = torchvision.utils.make_grid(images_to_log, nrow=6)
    return img_grid