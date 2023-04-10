import torch
import torch.nn.functional as F

from ..base_models import LitBaseCls
from .ts_transformer import TSTransformerEncoderClassiregressor

default_config = {
    'feat_dim': 6,
    'max_len': 160,
    'd_model': 64,
    'n_heads': 8,
    'num_layers': 8,
    'dim_feedforward': 256,
    'num_classes': 8,
    'dropout': 0.2,
    'pos_encoding': 'fixed',
    'activation': 'gelu',
    'norm': 'LayerNorm',
    'freeze': False
}


class LitTSTransformerClassifier(LitBaseCls):
    def __init__(self, config=default_config) -> None:
        super().__init__(config['num_classes'])

        self.config = config
        self.model_name = 'TSTransformerEncoderClassiregressor'

        self.model = TSTransformerEncoderClassiregressor(
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
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.model(x)
        val_loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("val/loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.model(x)
        val_loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("test/loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

        _, pred = torch.max(y_hat, dim=1)

        self.calculate_metrics(pred, y)
        self.log_all_metrics()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return [optimizer], [lr_scheduler]
