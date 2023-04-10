import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ..base_models import LitBaseCls
from .tcn import TCNClassifier


default_config = {
    'input_channels': 6,
    'levels': 8,
    'num_hid': 25,
    'num_classes': 8,
    'kernel_size': 2,
    'dropout': 0.2
}


class LitTCNClassifier(LitBaseCls):
    def __init__(self, config=default_config):
        super().__init__(config['num_classes'])

        self.config = config
        self.model_name = 'TCNClassifier'

        self.model = TCNClassifier(
            config['input_channels'],
            [config['num_hid']] * config['levels'],
            config['num_classes'],
            config['kernel_size'],
            config['dropout'])

    def training_step(self, batch, batch_index):
        x, _, y = batch
        x = x.permute(0, 2, 1)
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        x = x.permute(0, 2, 1)
        y_hat = self.model(x)
        val_loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("val/loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, _, y = batch
        x = x.permute(0, 2, 1)
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
