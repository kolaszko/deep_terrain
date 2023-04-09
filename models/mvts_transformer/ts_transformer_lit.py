import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from neptune.types import File
from sklearn.metrics import ConfusionMatrixDisplay

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


class LitTSTransformerClassifier(pl.LightningModule):
    def __init__(self, config=default_config) -> None:
        super().__init__()

        self.config = config
        self.model_name = 'TSTransformerEncoderClassiregressor'

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

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=config['num_classes'])
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=config['num_classes'], average='macro')
        self.precision = torchmetrics.Precision(task='multiclass', num_classes=config['num_classes'], average='macro')
        self.recall = torchmetrics.Recall(task='multiclass', num_classes=config['num_classes'], average='macro')
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            task='multiclass', num_classes=config['num_classes'])

    def training_step(self, batch, batch_index):
        x, _, y = batch
        y_hat = self.encoder(x)
        loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.encoder(x)
        val_loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("val/loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.encoder(x)
        val_loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("test/loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

        _, pred = torch.max(y_hat, dim=1)

        self.calculate_metrics(pred, y)
        self.log_all_metrics()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return [optimizer], [lr_scheduler]

    def calculate_metrics(self, pred, target):
        self.accuracy(pred, target)
        self.f1_score(pred, target)
        self.recall(pred, target)
        self.precision(pred, target)
        self.confusion_matrix(pred, target)

    def log_all_metrics(self):
        disp = ConfusionMatrixDisplay(self.confusion_matrix.compute().cpu().numpy())
        disp.plot(cmap=plt.cm.get_cmap("Blues"), xticks_rotation='vertical')
        self.logger.experiment["test/confusion_matrix"].upload(File.as_image(disp.figure_))

        self.log('test/accuracy', self.accuracy, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/f1_score', self.f1_score, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/recall', self.recall, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/precision', self.precision, on_epoch=True, on_step=False, prog_bar=True)
