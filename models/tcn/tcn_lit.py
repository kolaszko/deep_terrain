import torch
import torch.nn.functional as F

from ..base_models import LitBaseCls
from .tcn import TCNClassifier


class LitTCNClassifier(LitBaseCls):
    def __init__(self, config=None):
        super().__init__(config['num_classes'])

        self.config = config
        if self.config == None:
            self.config = LitTCNClassifier.get_default_config()

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

        self.calculate_val_metrics(y_hat, y)
        self.log_all_val_metrics()

    def test_step(self, batch, batch_idx):
        x, _, y = batch
        x = x.permute(0, 2, 1)
        y_hat = self.model(x)
        val_loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("test/loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

        _, pred = torch.max(y_hat, dim=1)

        self.calculate_test_metrics(pred, y)
        self.log_all_test_metrics()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return [optimizer], [lr_scheduler]

    @staticmethod
    def get_default_config():
        return {
            'input_channels': 6,
            'levels': 16,
            'num_hid': 305,
            'num_classes': 8,
            'kernel_size': 6,
            'dropout': 0.2
        }

    @classmethod
    def fromOptunaTrial(cls, trial):
        config = cls.get_default_config()
        config['levels'] = trial.suggest_int('levels', 1, 16, step=3)
        config['num_hid'] = trial.suggest_int('num_hid', 5, 305, step=50)
        config['kernel_size'] = trial.suggest_int('kernel_size', 2, 6, step=1)
        config['dropout'] = trial.suggest_float('dropout', 0.1, 0.8, step=0.1)

        return cls(config=config)
