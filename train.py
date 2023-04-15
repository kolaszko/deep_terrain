import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.tuner.tuning import Tuner

from data import LitHapticDataset
from models import (LitMLSTMfcnClassifier, LitTCNClassifier,
                    LitTSTransformerClassifier)


def objective(args):
    logger = NeptuneLogger(
        project="PPI/terrain-classification",
        api_token=os.getenv('NEPTUNE_API_TOKEN'),
        tags=["classification", "playground"],
        log_model_checkpoints=False)

    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=1, callbacks=[early_stop_callback, lr_monitor], logger=logger, log_every_n_steps=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)

    model = LitTSTransformerClassifier()
    data = LitHapticDataset(args.dataset_path, 128)

    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, datamodule=data)

    print(data.batch_size)

    logger.experiment['model'] = model.model_name
    logger.experiment['hyperparams'] = model.config

    trainer.fit(model, data)
    trainer.test(datamodule=data)

    logger.experiment.stop()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)

    args, _ = parser.parse_known_args()
    objective(args)
