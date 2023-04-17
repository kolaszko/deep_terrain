import torch
import torch.nn.functional as F

from ..base_models import LitBaseCls, LitBaseRegressor
from .ts_transformer import TSTransformerEncoderClassiregressor


class LitTSTransformerClassifier(LitBaseCls):
    def __init__(self, config):
        super().__init__(self.config['num_classes'])
        self.config = config

        self.model_name = 'TSTransformerEncoderClassiregressor'

        self.model = TSTransformerEncoderClassiregressor(
            self.config['feat_dim'],
            self.config['max_len'],
            self.config['d_model'],
            self.config['n_heads'],
            self.config['num_layers'],
            self.config['dim_feedforward'],
            self.config['num_classes'],
            dropout=self.config['dropout'],
            pos_encoding=self.config['pos_encoding'],
            activation=self.config['activation'],
            norm=self.config['norm'],
            freeze=self.config['freeze'])

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

        self.calculate_val_metrics(y_hat, y)
        self.log_all_val_metrics()

    def test_step(self, batch, batch_idx):
        x, _, y = batch
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
            'norm': 'LayerNorm',
            'freeze': False
        }

    @classmethod
    def fromOptunaTrial(cls, trial):
        config = cls.get_default_config()
        config['d_model'] = 2 ** trial.suggest_int('d_model', 5, 7, step=1)
        config['n_heads'] = 2 ** trial.suggest_int('num_lstm_out', 2, 4, step=1)
        config['num_layers'] = 2 ** trial.suggest_int('num_layers', 2, 4, step=1)
        config['dim_feedforward'] = 2 ** trial.suggest_int('dim_feedforward', 8, 9, step=1)
        config['pos_encoding'] = trial.suggest_categorical('pos_encoding', ['fixed', 'learnable'])
        config['activation'] = trial.suggest_categorical('activation', ['gelu', 'relu'])
        config['norm'] = trial.suggest_categorical('norm', ['LayerNorm', 'BatchNorm'])
        config['dropout'] = trial.suggest_float('dropout', 0.1, 0.8, step=0.1)

        return cls(config=config)
