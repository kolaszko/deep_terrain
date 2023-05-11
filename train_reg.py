import os
from argparse import ArgumentParser
from itertools import combinations

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.tuner.tuning import Tuner

from data import LitHapticDataset
from models import (LitMLSTMfcnClassifier,
                    LitTSTransformerClassifier, LitMLSTMfcnRegressor, LitTSTransformerRegressor,
                    LitHAPTRClassifier, LitHAPTRRegressor)


def train_cls(args, algorithm):
    logger = NeptuneLogger(
        project='PPI/moist-reg' if args.moist else "PPI/friction-regression",
        api_token=os.getenv('NEPTUNE_API_TOKEN'),
        tags=["initial-classification", "hyperparam"],
        log_model_checkpoints=False)

    model_checkpoint = ModelCheckpoint(monitor='val/accuracy', mode='max', save_top_k=1)
    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[
        EarlyStopping(monitor="val/accuracy", min_delta=0.00, patience=50, verbose=True, mode="max"),
        LearningRateMonitor(logging_interval='epoch'),
        model_checkpoint],
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=1,
        enable_progress_bar=True)

    data = LitHapticDataset(args.dataset_path, args.batch_size, args.moist)

    model_config = algorithm['cls'].get_default_config()
    model_config['num_classes'] = data.num_classes
    model_config['max_len'] = data.max_len

    model = algorithm['cls'](model_config)

    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, datamodule=data)

    logger.experiment['model'] = model.model_name
    logger.experiment['hyperparams'] = model.config
    logger.experiment['batch_size'] = data.batch_size

    trainer.fit(model, data)
    logger.experiment.stop()

    return model_checkpoint.best_model_path, model_config


def train_reg(args, algorithm, exclude_classes, cls_ckpt_path, cls_config):
    logger = NeptuneLogger(
        project='PPI/moist-reg' if args.moist else "PPI/friction-regression",
        api_token=os.getenv('NEPTUNE_API_TOKEN'),
        tags=["regression", "combinations", "best", "v2"],
        log_model_checkpoints=False)

    model_checkpoint = ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=1)
    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[
        EarlyStopping(monitor="val/loss", min_delta=0.00, patience=20, verbose=True, mode="min"),
        LearningRateMonitor(logging_interval='epoch'),
        model_checkpoint],
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=1)

    data = LitHapticDataset(args.dataset_path, args.batch_size, args.moist, cls=False, exclude_classes=exclude_classes)

    model_config = algorithm['cls'].get_default_config()
    model_config['num_classes'] = data.num_classes
    model_config['max_len'] = data.max_len

    model = algorithm['reg'](model_config)
    model.load_cls_state(cls_ckpt_path, cls_config)

    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, datamodule=data)
    model.set_max_min(data.max_c, data.min_c)

    logger.experiment['model'] = model.model_name
    logger.experiment['hyperparams'] = model.config
    logger.experiment['batch_size'] = data.batch_size
    logger.experiment['excluded_classes'] = str(exclude_classes)

    trainer.fit(model, data)
    trainer.test(datamodule=data, ckpt_path='best')

    logger.experiment.stop()


def task(args, algorithm):
    n_classes = 6 if args.moist else 8
    classes = [i for i in range(n_classes)]
    r = 1 if args.moist else 2

    exclude_classes = list(combinations(classes, r))
    exclude_classes.insert(0, ())

    print(exclude_classes)

    cls_ckpt, cls_config = train_cls(args, algorithm)

    for ex in exclude_classes:
        train_reg(args, algorithm, ex, cls_ckpt, cls_config)


def pipeline(args):
    algorithms = []

    if args.mlstm_fcn:
        algorithms.append(
            {
                'cls': LitMLSTMfcnClassifier,
                'reg': LitMLSTMfcnRegressor
            }
        )

    if args.ts_transformer:
        algorithms.append(
            {
                'cls': LitTSTransformerClassifier,
                'reg': LitTSTransformerRegressor
            }
        )
    
    if args.haptr:
        algorithms.append(
            {
                'cls': LitHAPTRClassifier,
                'reg': LitHAPTRRegressor
            }
        )

    print(algorithms)

    for algo in algorithms:
        task(args, algo)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/home/mikolaj/Datasets/moist/moist.txt')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-epochs', type=int, default=1)
    parser.add_argument('--mlstm-fcn', action='store_true')
    parser.add_argument('--ts-transformer', action='store_true')
    parser.add_argument('--haptr', action='store_false')
    parser.add_argument('--moist', action='store_true')

    args, _ = parser.parse_known_args()
    pipeline(args)
