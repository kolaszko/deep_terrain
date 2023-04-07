import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ts_transformer import TSTransformerEncoderClassiregressor

default_config = {
    'feat_dim': 6,
    'max_len': 160,
    'd_model': 64,
    'n_heads': 8,
    'num_layers': 3,
    'dim_feedforward': 256,
    'num_classes': 8,
    'dropout': 0.1,
    'pos_encoding': 'fixed',
    'activation': 'gelu',
    'norm': 'BatchNorm',
    'freeze': False
}


class LitTSTransformerClassifier(pl.LightningModule):
    def __init__(self, config=default_config) -> None:
        super().__init__()

        self.encoder = TSTransformerEncoderClassiregressor(
            config['feat_dim'],
            config['max_len'],
            config['d_model'],
            config['n_heads'],
            config['num_layers'],
            config['dim_feedforward'],
            config['num_classes'],
            dropout=config['dropout'],
            pos_encoding=config['pos_encoding'],
            activation=config['activation'],
            norm=config['norm'],
            freeze=config['freeze'])

    def training_step(self, batch, batch_index):
        x, _, y = batch
        y_hat = self.encoder(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.encoder(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val/loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
