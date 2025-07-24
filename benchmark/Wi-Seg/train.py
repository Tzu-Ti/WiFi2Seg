import sys
sys.path.append("../..")  # Adjust the path to include the parent directory
from dataset import CSI2MaskDataModule
from models import WiSegUNet
import utils

from torch.utils.data import DataLoader
from torch import nn
import torch

from torchmetrics import KLDivergence
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.segmentation import DiceScore

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import yaml
import argparse
import einops

class WiSegUNetLightning(LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.save_hyperparameters()

        # training configurations
        self.training_config = training_config = configs['Training']

        # data configurations
        data_configs = configs['Data']
        length = data_configs['length']  # 51
        RxTx_num = data_configs['RxTx_num']  # 6
        in_channels = data_configs['subcarrier_num']  # 2025

        # Model configurations
        model_configs = configs['Model']
        self.model = WiSegUNet(in_channels=length * in_channels, reduced_channels=model_configs['latent_dim'])
        
       # Loss functions
        self.BCE = nn.BCEWithLogitsLoss(reduction='mean')

        # Metrics
        self.threshold = 0.3
        self.IoU = BinaryJaccardIndex(threshold=self.threshold)
        self.DICE = DiceScore(num_classes=1, average="micro")
        

    def forward(self, csi):
        csi = einops.rearrange(csi, 'b l (rx antennas) c -> b (l c) rx antennas', rx=3, antennas=2)
        seg = self.model(csi)  # Forward pass through the model

        return seg


    def training_step(self, batch, batch_idx):
        csi, mask = batch
        seg = self.forward(csi)

        bce_loss = self.BCE(seg, mask)

        total_loss = bce_loss

        self.log_dict({
            'train/bce_loss': bce_loss,
            'train/total_loss': total_loss
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # log image
        if batch_idx % 1000 == 0:
            img_grid = utils.make_img_grid(mask, seg)
            self.logger.experiment.add_images('train/images', img_grid, self.global_step, dataformats="CHW")

        return total_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        csi, mask = batch
        seg = self.forward(csi)

        seg = torch.sigmoid(seg)  # Apply sigmoid to get probabilities
        seg = (seg > 0.5).float()  # Binarize the output based on the threshold
        dice = self.DICE(seg, mask)
        iou = self.IoU(seg.float(), mask.long())

        if dataloader_idx == 0:
            prefix = 'seen'
        else:
            prefix = 'unseen'

        self.log_dict({
            f'val/{prefix}/dice': dice,
            f'val/{prefix}/iou': iou,
        }, on_step=True, prog_bar=True, logger=True)

        # log image
        if batch_idx % 1000 == 0:
            img_grid = utils.make_img_grid(mask, seg)
            self.logger.experiment.add_images(f'val/{prefix}/images', img_grid, self.global_step, dataformats="CHW")

    def test_step(self, batch, batch_idx):
        csi, mask = batch
        seg = self.forward(csi)

        seg = torch.sigmoid(seg)  # Apply sigmoid to get probabilities
        seg = (seg > 0.5).float()  # Binarize the output based on the threshold
        dice = self.DICE(seg, mask)
        iou = self.IoU(seg.float(), mask.long())

        self.log_dict({
            'test/dice': dice,
            'test/iou': iou,
        }, on_step=True, prog_bar=True, logger=True)

        # log image
        if batch_idx % 1000 == 0:
            img_grid = utils.make_img_grid(mask, seg)
            self.logger.experiment.add_images('test/images', img_grid, self.global_step, dataformats="CHW")

    def configure_optimizers(self):
        lr = self.training_config['lr']
        opt = utils.Optimizer(
            optimizer=self.training_config['optimizer'],
            params=self.parameters(),
            lr=lr
        )
        return opt
        
def main(args):
    # Load model-specific configuration from YAML
    with open(args.configs, 'r') as f:
        configs = yaml.load(f, Loader=yaml.CLoader)

    # Initialize LightningModule
    model = WiSegUNetLightning(configs=configs)

    # Setup configuration
    training_config = configs['Training']

    # Setup data module
    dm = CSI2MaskDataModule(configs=configs)

    # Setup Tensorboard logger
    logger = TensorBoardLogger("lightning_logs", 
                               name="WiSeg",
                               version=args.version,)
    
    # Initialize PyTorch Lightning Trainer
    if args.mode == 'train':
        trainer = Trainer(
            max_epochs=training_config['epochs'],
            logger=logger,
            precision='16-mixed',
            log_every_n_steps=100,
            num_sanity_val_steps=5,
            # fast_dev_run=True
        )
        # resume training
        if args.ckpt_path:
            trainer.fit(model, dm, ckpt_path=args.ckpt_path)
        # regular training
        trainer.fit(model, dm)
    elif args.mode == 'test':
        trainer = Trainer(logger=logger, num_nodes=1)
        trainer.test(model, datamodule=dm, ckpt_path=args.ckpt_path)

def parser():
    parser = argparse.ArgumentParser(description="Train WiFi2Seg model with PyTorch Lightning")
    parser.add_argument('--configs', type=str, help='Path to the configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'test'],)
    parser.add_argument('--ckpt_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('-v', '--version', type=str, default='test', help='The name of this procedure')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser()
    main(args)