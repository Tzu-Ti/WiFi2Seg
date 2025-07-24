from datas.dataset import CSI2MaskDataset, CSI2MaskDataModule
from models import embedding, encoder, modules, VAE
import utils
import loss

from torch import nn
import torch

from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.segmentation import DiceScore

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import yaml
import argparse

class CSIEncoderLightning(LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.save_hyperparameters()

        # training configurations
        self.training_config = training_config = configs['Training']

        # data configurations
        data_configs = configs['Data']
        length = data_configs['length'] # 51
        RxTx_num = data_configs['RxTx_num'] # 6
        in_channels = data_configs['subcarrier_num'] # 2025

        # CSI Encoder configurations
        ha_configs = configs['HybridAttention']
        d_model = ha_configs['d_model'] # 256
        n_head = ha_configs['n_head'] # 8
        kernel_sizes = ha_configs['kernel_sizes'] # [3, 5, 7]
        mask = ha_configs['mask'] # None
        num_layers = ha_configs['num_layers'] # 3
        self.use_reverse = use_reverse = ha_configs['use_reverse']

        # VAE configurations
        vae_configs = configs['VAE']
        vae_model_configs = vae_configs['Model']
        latent_dim = vae_model_configs['latent_dim']
        vae_enc_channels = vae_model_configs['enc_channels']
        vae_dec_channels = vae_model_configs['dec_channels']
        vae_ckpt = training_config['vae_ckpt_path'] # Path to the VAE checkpoint


        self.amp_embedding = embedding.GaussianRangeEmbedding(
            length=length, RxTx_num=RxTx_num, in_channels=in_channels, d_model=d_model
        )
        self.pha_embedding = embedding.GaussianRangeEmbedding(
            length=length, RxTx_num=RxTx_num, in_channels=in_channels, d_model=d_model
        )

        self.hybrid_attention_layer = encoder.HybridAttentionLayer(
            length=length, RxTx_num=RxTx_num,
            d_model=d_model, n_head=n_head,
            kernel_sizes=kernel_sizes,
            mask=mask,
            use_reverse=use_reverse
        )

        self.hybrid_attention = encoder.HybridAttention(
            layer=self.hybrid_attention_layer,
            num_layers=num_layers
        )

        # Aggregation configuraions
        aggregation_configs = configs['Aggregation']
        n_head = aggregation_configs['n_head'] # 8
        self.aggregation = modules.CrossAggregationBlock(
            length=length, RxTx_num=RxTx_num,
            d_model=d_model, latent_dim=latent_dim, n_head=n_head
        )

        # VAE
        self.fc_mu = nn.Linear(length*latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(length*latent_dim, latent_dim)
        self.vae = VAE.VAE(enc_channels=vae_enc_channels,
                           dec_channels=vae_dec_channels,
                           latent_dim=latent_dim)
        if vae_ckpt is not None:
            print("> Loading VAE checkpoint from:", vae_ckpt)
            self.vae = utils.load_ckpt(self.vae, vae_ckpt, freeze=True)

        # Loss functions
        self.BCE = nn.BCEWithLogitsLoss(reduction='mean')
        self.DICE = DiceScore(num_classes=1, average="micro")
        self.KL = loss.KLLoss()
        self.CL = loss.ContrastiveLoss()

        # Metrics
        self.threshold = 0.3
        self.IoU = BinaryJaccardIndex(threshold=self.threshold)

        # Loss weights
        self.W_BCE = 1
        self.W_KL = 1e-3
        self.W_DICE = 2
        self.W_CL = 1


    def forward(self, amp, pha):
        # embedding
        amp = self.amp_embedding(amp)
        pha = self.pha_embedding(pha)
        # hybrid attention
        if self.use_reverse:
            amp, pha, amp_rev, pha_rev = self.hybrid_attention(amp, pha)
        else:
            amp, pha = self.hybrid_attention(amp, pha)
        # aggregation
        feat = self.aggregation(amp, pha)
        # vae
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        z = self.vae.reparameterize(mu, logvar)
        recon = self.vae.decoder(z)

        if self.use_reverse:
            return recon, mu, logvar, amp_rev, pha_rev
        else:
            return recon, mu, logvar

        
    def training_step(self, batch, batch_idx):
        [amp, pha, mask], [another_amp, another_pha], label = batch
        if self.use_reverse:
            out, mu, logvar, amp_rev, pha_rev = self.forward(amp, pha)
            _, _, _, another_amp_rev, another_pha_rev = self.forward(another_amp, another_pha)
        else:
            out, mu, logvar = self.forward(amp, pha)

        bce_loss = self.BCE(out, mask)
        dice_loss = 1 - self.DICE(torch.sigmoid(out), mask)
        kl_loss = self.KL(mu, logvar)
        # kl_loss = 0

        if self.use_reverse:
            cl_loss = self.CL(anchor=[amp_rev, pha_rev],
                              feature=[another_amp_rev, another_pha_rev],
                              label=label)
            total_loss = bce_loss * self.W_BCE + dice_loss * self.W_DICE + kl_loss * self.W_KL + cl_loss * self.W_CL
        else:
            total_loss = bce_loss * self.W_BCE + dice_loss * self.W_DICE + kl_loss * self.W_KL

        self.log_dict({
            'train/total_loss': total_loss,
            'train/bce': bce_loss,
            'train/dice': dice_loss,
            'train/kl': kl_loss,
            'train/lr': self.optimizers().param_groups[0]['lr'],
            'train/cl': cl_loss if self.use_reverse else 0
        }, on_step=True, prog_bar=True, logger=True)

        # log image
        if batch_idx % 1000 == 0:
            out = torch.sigmoid(out)
            out = (out > 0.5).float()  # Binarize the output based on the threshold
            img_grid = utils.make_img_grid(mask, out)
            self.logger.experiment.add_images('train/images', img_grid, self.global_step, dataformats="CHW")
        
        return total_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        amp, pha, mask = batch
        if self.use_reverse:
            out, mu, logvar, _, _ = self.forward(amp, pha)
        else:
            out, mu, logvar = self.forward(amp, pha)
        
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
        }, on_step=True, prog_bar=True, logger=True)

        # log image
        if batch_idx % 1000 == 0:
            img_grid = utils.make_img_grid(mask, out)
            self.logger.experiment.add_images(f'val/{prefix}/images', img_grid, self.global_step, dataformats="CHW")

    def test_step(self, batch, batch_idx):
        amp, pha, mask = batch
        if self.use_reverse:
            out, mu, logvar, _, _ = self.forward(amp, pha)
        else:
            out, mu, logvar = self.forward(amp, pha)
        
        dice = self.DICE(out, mask)
        iou = self.IoU(out.float(), mask.long())

        self.log_dict({
            f'test/dice': dice,
            f'test/iou': iou,
        }, on_step=True, prog_bar=True, logger=True)

        # log image
        if batch_idx % 1000 == 0:
            img_grid = utils.make_img_grid(mask, out)
            self.logger.experiment.add_images(f'test/images', img_grid, self.global_step, dataformats="CHW")
        
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
    model = CSIEncoderLightning(configs=configs)

    # Setup configuration
    training_config = configs['Training']

    # Setup data module
    dm = CSI2MaskDataModule(dataset_class=CSI2MaskDataset, configs=configs)

    # Setup Tensorboard logger
    logger = TensorBoardLogger("lightning_logs", 
                               name="WiFi2Seg",
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
    parser = argparse.ArgumentParser(description="Train WiFi2Seg model with PyTorch Lightning")
    parser.add_argument('--configs', type=str, help='Path to the configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'test'],)
    parser.add_argument('--ckpt_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('-v', '--version', type=str, default='test', help='The name of this procedure')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser()
    main(args)