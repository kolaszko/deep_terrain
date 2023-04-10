import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger

from data import LitHapticDataset
from models import LitTSTransformerClassifier, LitTCNClassifier


def objective(args):
    logger = NeptuneLogger(
        project="PPI/terrain-classification",
        api_token=os.getenv('NEPTUNE_API_TOKEN'),
        tags=["classification", "playground"],
        log_model_checkpoints=False)

    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=250, callbacks=[early_stop_callback, lr_monitor], logger=logger)

    # model = LitTSTransformerClassifier()
    model = LitTCNClassifier()
    data = LitHapticDataset(args.dataset_path, 128)

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
