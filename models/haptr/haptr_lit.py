import torch
import torch.nn.functional as F

from ..base_models import LitBaseCls, LitBaseRegressor
from .haptr import HAPTR_ModAtt


class LitHAPTRClassifier(LitBaseCls):
    def __init__(self, config):
        super().__init__(config['num_classes'])
        self.config = config

        self.model_name = 'HAPTRClassifier'
        self.model = HAPTR_ModAtt(
            self.config['num_classes'],
            self.config['projection_dim'],
            self.config['max_len'],
            self.config['nheads'],
            self.config['num_encoder_layers'],
            self.config['feed_forward'],
            self.config['dropout'],
            self.config['dim_modalities'],
            self.config['num_modalities']
        )
    
    def training_step(self, batch, batch_index):
        x, _, y = batch
        x = self.__prepare_input(x)

        y_hat, _ = self.model(x)
        loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        x = self.__prepare_input(x)

        y_hat, _ = self.model(x)
        val_loss = F.cross_entropy(y_hat, y, torch.tensor(
            self.trainer.datamodule.weights, dtype=torch.float, device=self.device))
        self.log("val/loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

        self.calculate_val_metrics(y_hat, y)
        self.log_all_val_metrics()

    def test_step(self, batch, batch_idx):
        x, _, y = batch
        x = self.__prepare_input(x)

        y_hat, _ = self.model(x)
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
    
    def __prepare_input(self, x):
        sig = []
        start_idx = 0
        for _, mod_dim in zip([0, 1], self.config['dim_modalities']):
            end_idx = start_idx + mod_dim
            sig.append(x[..., start_idx:end_idx])
            start_idx = end_idx
        
        return sig

    @staticmethod
    def get_default_config():
        return {
            'num_classes': 8,
            'projection_dim': 16,
            'max_len': 160,
            'nheads': 8,
            'num_encoder_layers': 8,
            'feed_forward': 128,
            'dropout': 0.1,
            'dim_modalities': [3, 3],
            'num_modalities': 2
        }
    
class LitHAPTRRegressor(LitBaseRegressor):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model_name = 'HAPTRRegressor'

        self.model = HAPTR_ModAtt(
            self.config['num_classes'],
            self.config['projection_dim'],
            self.config['max_len'],
            self.config['nheads'],
            self.config['num_encoder_layers'],
            self.config['feed_forward'],
            self.config['dropout'],
            self.config['dim_modalities'],
            self.config['num_modalities']
        )
    
    def training_step(self, batch, batch_index):
        x, y, _ = batch
        x = self.__prepare_input(x)

        y_hat = F.gelu(self.model(x)[0])
        y_hat = torch.squeeze(y_hat)

        loss = F.mse_loss(y_hat, y)
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        x = self.__prepare_input(x)

        y_hat = F.gelu(self.model(x)[0])
        y_hat = torch.squeeze(y_hat)

        val_loss = F.mse_loss(y_hat, y)
        self.log("val/loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

        self.calculate_val_metrics(y_hat, y)
        self.log_all_val_metrics()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        x = self.__prepare_input(x)

        y_hat = F.gelu(self.model(x)[0])
        y_hat = torch.squeeze(y_hat)

        val_loss = F.mse_loss(y_hat, y)
        self.log("test/loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

        self.calculate_test_metrics(y_hat, y)
        self.log_all_test_metrics()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        return [optimizer], [lr_scheduler]
    
    def __prepare_input(self, x):
        sig = []
        start_idx = 0
        for _, mod_dim in zip([0, 1], self.config['dim_modalities']):
            end_idx = start_idx + mod_dim
            sig.append(x[..., start_idx:end_idx])
            start_idx = end_idx
        
        return sig
    
    def load_cls_state(self, cls_ckpt_path, config):
        classifier = LitHAPTRClassifier.load_from_checkpoint(
            cls_ckpt_path, config=config)

        cls_dict = classifier.model.state_dict()
        del cls_dict['mlp_head.2.weight']
        del cls_dict['mlp_head.2.bias']

        self.model.load_state_dict(cls_dict, strict=False)

    @staticmethod
    def get_default_config():
        return {
            'num_classes': 1,
            'projection_dim': 16,
            'max_len': 160,
            'nheads': 8,
            'num_encoder_layers': 8,
            'feed_forward': 128,
            'dropout': 0.1,
            'dim_modalities': [3, 3],
            'num_modalities': 2
        }