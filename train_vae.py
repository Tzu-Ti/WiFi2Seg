from datas.dataset import MaskDataset, CSI2MaskDataModule
from models import VAE
import utils
import loss

import torch
import torchvision
from torch import nn

from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.segmentation import DiceScore

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import yaml
import argparse

class VAELightning(LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.save_hyperparameters()

        # VAE configurations
        vae_configs = configs['VAE']

        # training configurations
        self.training_config = training_config = vae_configs['Training']

        # model configurations
        self.model_config = vae_configs['Model']
        self.net = VAE.VAE(enc_channels=self.model_config['enc_channels'],
                           dec_channels=self.model_config['dec_channels'],
                           latent_dim=self.model_config['latent_dim'])

        # Loss functions
        self.BCE = nn.BCEWithLogitsLoss(reduction='mean')
        self.DICE = DiceScore(num_classes=1, average="micro")
        self.KL = loss.KLLoss()

        # Metrics
        self.threshold = 0.3
        self.IoU = BinaryJaccardIndex(threshold=self.threshold)

        # Loss weights
        self.W_BCE = 1
        self.W_KL = 1e-3
        self.W_DICE = 2

    def forward(self, mask):
        out, mu, logvar = self.net(mask)
        return out, mu, logvar
    
    def training_step(self, batch, batch_idx):
        mask = batch
        out, mu, logvar = self.forward(mask)

        bce_loss = self.BCE(out, mask)
        dice_loss = 1 - self.DICE(torch.sigmoid(out), mask)
        kl_loss = self.KL(mu, logvar)
        
        total_loss = bce_loss * self.W_BCE + dice_loss * self.W_DICE + kl_loss * self.W_KL

        self.log_dict({
            'train/total_loss': total_loss,
            'train/bce': bce_loss,
            'train/dice': dice_loss,
            'train/kl': kl_loss,
            'train/lr': self.optimizers().param_groups[0]['lr'],
        }, on_step=True, prog_bar=True, logger=True)

        # log image
        if batch_idx % 1000 == 0:
            out = torch.sigmoid(out)
            out = (out > 0.5).float()  # Binarize the output based on the threshold
            img_grid = utils.make_img_grid(mask, out)
            self.logger.experiment.add_images('train/images', img_grid, self.global_step, dataformats="CHW")

        return total_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        mask = batch
        out, mu, logvar = self.forward(mask)

        out = torch.sigmoid(out)
        out = (out > 0.5).float()
        dice = self.DICE(out, mask)
        iou = self.IoU(out.float(), mask.long())

        if dataloader_idx == 0:
            prefix = 'seen'
        else:
            prefix = 'unseen'

        self.log_dict({
            f'val/{prefix}/dice': dice,
            f'val/{prefix}/iou': iou,
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

        # log image
        if batch_idx % 1000 == 0:
            img_grid = utils.make_img_grid(mask, out)
            self.logger.experiment.add_images(f'val/{prefix}/images', img_grid, self.global_step, dataformats="CHW")
    
    def test_step(self, batch, batch_idx):
        mask = batch
        out, mu, logvar = self.forward(mask)

        out = torch.sigmoid(out)
        out = (out > 0.5).float()
        dice = self.DICE(out, mask)
        iou = self.IoU(out.float(), mask.long())

        self.log_dict({
            'test/dice': dice,
            'test/IoU': iou,
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # log image
        if batch_idx % 1000 == 0:
            img_grid = utils.make_img_grid(mask, out)
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
    model = VAELightning(configs=configs)

    # Setup configuration
    training_config = configs['Training']

    # Setup data module
    dm = CSI2MaskDataModule(dataset_class=MaskDataset, configs=configs)

    # Setup Tensorboard logger
    logger = TensorBoardLogger("lightning_logs", 
                               name="VAE",
                               version=args.version)
    
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
        trainer.fit(model, dm)
    elif args.mode == 'test':
        trainer = Trainer(logger=logger, num_nodes=1)
        trainer.test(model, datamodule=dm, ckpt_path=args.ckpt_path)

def parser():
    parser = argparse.ArgumentParser(description="Train VAE of WiFi2Seg model with PyTorch Lightning")
    parser.add_argument('--configs', type=str, help='Path to the configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'],)
    parser.add_argument('--ckpt_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('-v', '--version', type=str, default='test', help='The name of this procedure')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser()
    main(args)

    

    
