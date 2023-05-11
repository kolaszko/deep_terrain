import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import *


class LitHapticDataset(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, is_moist=False, split_size=0.2, cls=True, exclude_classes=()) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.cls = cls
        self.exclude_classes = exclude_classes
        self.split_size = split_size
        self.is_moist = is_moist
        if self.is_moist:
            self.num_classes = 6
            self.max_len = 150
        else:
            self.num_classes = 8
            self.max_len = 160

    def setup(self, stage: str) -> None:
        if not self.is_moist:
            train_ds, val_ds, test_ds, weights = get_putany_cls_dataset(
                self.data_dir, self.split_size) if self.cls else get_putany_regression_dataset(
                self.data_dir, self.split_size, self.exclude_classes)
        else:
            train_ds, val_ds, test_ds, weights = get_moist_cls_dataset(
                self.data_dir, self.split_size) if self.cls else get_moist_regression_dataset(
                self.data_dir, self.split_size, self.exclude_classes)

        self.train_ds = HapticDataset(train_ds)
        self.val_ds = HapticDataset(val_ds)
        self.test_ds = HapticDataset(test_ds)
        self.weights = weights

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
