import torch
import torch.nn.functional as F

from ..base_models import LitBaseCls
from .mlstm_fcn import MLSTMfcn

default_config = {
    'num_features': 6,
    'max_seq_len': 160,
    'num_lstm_layers': 1,
    'num_lstm_out': 128,
    'num_classes': 8,
    'conv1_nf': 128,
    'conv2_nf': 256,
    'conv3_nf': 128,
    'lstm_drop_p': 0.8,
    'fc_drop_p': 0.3
}


class LitMLSTMfcnClassifier(LitBaseCls):
    def __init__(self, config=default_config):
        super().__init__(config['num_classes'])

        self.config = config
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

    def test_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.model(x, [160] * x.shape[0])
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
