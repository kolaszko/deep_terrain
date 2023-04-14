import torch
import torch.nn.functional as F

from ..base_models import LitBaseCls
from .mlstm_fcn import MLSTMfcn


class LitMLSTMfcnClassifier(LitBaseCls):
    def __init__(self, config=None):
        super().__init__(config['num_classes'])

        self.config = config
        if self.config == None:
            self.config = LitMLSTMfcnClassifier.get_default_config()

        self.model_name = 'MLSTMfcnClassifier'

        self.model = MLSTMfcn(
            num_classes=config['num_classes'],
            max_seq_len=config['max_seq_len'],
            num_features=config['num_features'],
            num_lstm_out=config['num_lstm_out'],
            num_lstm_layers=config['num_lstm_layers'],
            conv1_nf=config['conv1_nf'],
            conv2_nf=config['conv2_nf'],
            conv3_nf=config['conv3_nf'],
            lstm_drop_p=config['lstm_drop_p'],
            fc_drop_p=config['fc_drop_p']
        )

    def training_step(self, batch, batch_index):
        x, _, y = batch
        y_hat = self.model(x, [160] * x.shape[0])
        loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.model(x, [160] * x.shape[0])
        val_loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("val/loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

        self.calculate_val_metrics(y_hat, y)
        self.log_all_val_metrics()

    def test_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.model(x, [160] * x.shape[0])
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
            'num_features': 6,
            'max_seq_len': 160,
            'num_lstm_layers': 2,
            'num_lstm_out': 128,
            'num_classes': 8,
            'conv1_nf': 128,
            'conv2_nf': 256,
            'conv3_nf': 128,
            'lstm_drop_p': 0.8,
            'fc_drop_p': 0.3
        }

    @classmethod
    def fromOptunaTrial(cls, trial):
        config = cls.get_default_config()
        config['num_lstm_layers'] = trial.suggest_categorical('num_lstm_layers', [1, 2, 4, 8])
        config['num_lstm_out'] = trial.suggest_categorical('num_lstm_out', [128, 256])
        config['conv1_nf'] = trial.suggest_categorical('conv1_nf', [64, 128, 256])
        config['conv2_nf'] = trial.suggest_categorical('conv2_nf', [128, 256, 512])
        config['conv3_nf'] = trial.suggest_categorical('conv3_nf', [128, 256, 512])
        config['lstm_drop_p'] = trial.suggest_float('lstm_drop_p', 0.1, 0.8)
        config['fc_drop_p'] = trial.suggest_float('fc_drop_p', 0.1, 0.8)

        return cls(config=config)
