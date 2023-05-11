import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_models import LitBaseCls, LitBaseRegressor
from .mlstm_fcn import MLSTMfcn


class LitMLSTMfcnClassifier(LitBaseCls):
    def __init__(self, config):
        super().__init__(config['num_classes'])

        self.config = config
        self.model_name = 'MLSTMfcnClassifier'

        self.model = MLSTMfcn(
            num_classes=config['num_classes'],
            max_seq_len=config['max_len'],
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
        y_hat = self.model(x, [self.config['max_len']] * x.shape[0])
        loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.model(x, [self.config['max_len']] * x.shape[0])
        val_loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("val/loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

        self.calculate_val_metrics(y_hat, y)
        self.log_all_val_metrics()

    def test_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.model(x, [self.config['max_len']] * x.shape[0])
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
            'max_len': 160,
            'num_lstm_layers': 8,
            'num_lstm_out': 256,
            'num_classes': 8,
            'conv1_nf': 256,
            'conv2_nf': 256,
            'conv3_nf': 512,
            'lstm_drop_p': 0.4,
            'fc_drop_p': 0.1
        }

    @classmethod
    def fromOptunaTrial(cls, trial):
        config = cls.get_default_config()
        config['num_lstm_layers'] = 2 ** trial.suggest_int('num_lstm_layers', 1, 3, step=1)
        config['num_lstm_out'] = 2 ** trial.suggest_int('num_lstm_out', 7, 8, step=1)
        config['conv1_nf'] = 2 ** trial.suggest_int('conv1_nf', 6, 8, step=1)
        config['conv2_nf'] = 2 ** trial.suggest_int('conv2_nf', 7, 9, step=1)
        config['conv3_nf'] = 2 ** trial.suggest_int('conv3_nf', 7, 9, step=1)
        config['lstm_drop_p'] = trial.suggest_float('lstm_drop_p', 0.1, 0.8, step=0.1)
        config['fc_drop_p'] = trial.suggest_float('fc_drop_p', 0.1, 0.8, step=0.1)

        return cls(config=config)


class LitMLSTMfcnRegressor(LitBaseRegressor):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_name = 'MLSTMfcnClassifier'

        self.model = MLSTMfcn(
            num_classes=1,
            max_seq_len=config['max_len'],
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
        x, y, _ = batch

        y_hat = F.gelu(self.model(x, [self.config['max_len']] * x.shape[0]))
        y_hat = torch.squeeze(y_hat)

        loss = F.mse_loss(y_hat, y)
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        y_hat = F.gelu(self.model(x, [self.config['max_len']] * x.shape[0]))
        y_hat = torch.squeeze(y_hat)

        val_loss = F.mse_loss(y_hat, y)
        self.log("val/loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

        self.calculate_val_metrics(y_hat, y)
        self.log_all_val_metrics()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch

        y_hat = F.gelu(self.model(x, [self.config['max_len']] * x.shape[0]))
        y_hat = torch.squeeze(y_hat)

        test_loss = F.mse_loss(y_hat, y)
        self.log("test/loss", test_loss, on_epoch=True, on_step=False, prog_bar=True)

        self.calculate_test_metrics(y_hat, y)
        self.log_all_test_metrics()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return [optimizer], [lr_scheduler]

    def load_cls_state(self, cls_ckpt_path, config):
        classifier = LitMLSTMfcnClassifier.load_from_checkpoint(
            cls_ckpt_path, config=config)

        cls_dict = classifier.model.state_dict()
        del cls_dict['fc.weight']
        del cls_dict['fc.bias']

        self.model.load_state_dict(cls_dict, strict=False)

    @staticmethod
    def get_default_config():
        return {
            'num_features': 6,
            'max_len': 160,
            'num_lstm_layers': 8,
            'num_lstm_out': 256,
            'num_classes': 1,
            'conv1_nf': 256,
            'conv2_nf': 256,
            'conv3_nf': 512,
            'lstm_drop_p': 0.4,
            'fc_drop_p': 0.1
        }

    @classmethod
    def fromOptunaTrial(cls, trial):
        config = cls.get_default_config()
        config['num_lstm_layers'] = 2 ** trial.suggest_int('num_lstm_layers', 1, 3, step=1)
        config['num_lstm_out'] = 2 ** trial.suggest_int('num_lstm_out', 7, 8, step=1)
        config['conv1_nf'] = 2 ** trial.suggest_int('conv1_nf', 6, 8, step=1)
        config['conv2_nf'] = 2 ** trial.suggest_int('conv2_nf', 7, 9, step=1)
        config['conv3_nf'] = 2 ** trial.suggest_int('conv3_nf', 7, 9, step=1)
        config['lstm_drop_p'] = trial.suggest_float('lstm_drop_p', 0.1, 0.8, step=0.1)
        config['fc_drop_p'] = trial.suggest_float('fc_drop_p', 0.1, 0.8, step=0.1)

        return cls(config=config)
