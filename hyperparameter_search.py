import os
from argparse import ArgumentParser

import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.tuner.tuning import Tuner

from data import LitHapticDataset
from models import (LitMLSTMfcnClassifier, LitTCNClassifier,
                    LitTSTransformerClassifier)


def objective(trial, args, algorithm):
    logger = NeptuneLogger(
        project="PPI/terrain-cls-hyperparam",
        api_token=os.getenv('NEPTUNE_API_TOKEN'),
        tags=["classification", "hyperparam", "trial"],
        log_model_checkpoints=False)

    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[
        PyTorchLightningPruningCallback(trial, monitor="val/accuracy"),
        LearningRateMonitor(logging_interval='epoch')], logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1, log_every_n_steps=2,
        enable_progress_bar=False)

    try:
        data = LitHapticDataset(args.dataset_path, args.batch_size)
        model = algorithm.fromOptunaTrial(trial)

        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=data)

        logger.experiment['model'] = model.model_name
        logger.experiment['hyperparams'] = model.config
        logger.experiment['batch_size'] = data.batch_size

        trainer.fit(model, data)
    finally:
        metric = trainer.callback_metrics["val/accuracy"].item()
        logger.experiment.stop()

    return metric


def rerun_best_trial(trial, args, algorithm):
    logger = NeptuneLogger(
        project="PPI/terrain-cls-hyperparam",
        api_token=os.getenv('NEPTUNE_API_TOKEN'),
        tags=["classification", "hyperparam", "best"],
        log_model_checkpoints=False)

    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[
        EarlyStopping(monitor="val/accuracy", min_delta=0.00, patience=50, verbose=True, mode="max"),
        LearningRateMonitor(logging_interval='epoch')],
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1, log_every_n_steps=1)

    try:
        data = LitHapticDataset(args.dataset_path, args.batch_size)
        model = algorithm.fromOptunaTrial(trial)

        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=data)

        logger.experiment['model'] = model.model_name
        logger.experiment['hyperparams'] = model.config
        logger.experiment['batch_size'] = data.batch_size

        trainer.fit(model, data)
        trainer.test(datamodule=data)
    finally:
        logger.experiment.stop()


def optuna_pipeline(args):

    algorithms = []
    if args.tcn:
        algorithms.append(LitTCNClassifier)

    if args.mlstm_fcn:
        algorithms.append(LitMLSTMfcnClassifier)

    if args.ts_transformer:
        algorithms.append(LitTSTransformerClassifier)

    print(algorithms)

    for algo in algorithms:
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(lambda trial: objective(trial, args, algo), gc_after_trial=True)
        rerun_best_trial(study.best_trial, args, algo)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/root/friction_classes.pickle')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-epochs', type=int, default=350)
    parser.add_argument('--tcn', action='store_true')
    parser.add_argument('--mlstm-fcn', action='store_true')
    parser.add_argument('--ts-transformer', action='store_true')

    args, _ = parser.parse_known_args()
    optuna_pipeline(args)
