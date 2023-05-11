import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchmetrics
from neptune.types import File
from sklearn.metrics import ConfusionMatrixDisplay


class LitBaseRegressor(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model_name = 'BaseRegressor'

        self.concordance_cc = torchmetrics.ConcordanceCorrCoef()
        self.cosine_similarity = torchmetrics.CosineSimilarity()
        self.explained_variance = torchmetrics.ExplainedVariance()
        self.kendall_tau = torchmetrics.KendallRankCorrCoef()
        self.log_cosh_error = torchmetrics.LogCoshError()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mape = torchmetrics.MeanAbsolutePercentageError()
        self.mse = torchmetrics.MeanSquaredError()
        self.msle = torchmetrics.MeanSquaredLogError()
        self.pearson_corrcoef = torchmetrics.PearsonCorrCoef()
        self.r2score = torchmetrics.R2Score()
        self.spearman_corrcoef = torchmetrics.SpearmanCorrCoef()
        self.smape = torchmetrics.SymmetricMeanAbsolutePercentageError()
        self.wmape = torchmetrics.WeightedMeanAbsolutePercentageError()

        self.val_mape = torchmetrics.MeanAbsolutePercentageError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

        self.max_c = 1
        self.min_c = 0

    def calculate_test_metrics(self, m_pred, m_target):
        pred = (m_pred * (self.max_c - self.min_c)) + self.min_c
        target = (m_target * (self.max_c - self.min_c)) + self.min_c

        self.concordance_cc(pred, target)
        self.cosine_similarity(pred, target)
        self.explained_variance(pred, target)
        self.kendall_tau(pred, target)
        self.log_cosh_error(pred, target)
        self.mae(pred, target)
        self.mape(pred, target)
        self.mse(pred, target)
        self.msle(pred, target)
        self.pearson_corrcoef(pred, target)
        self.r2score(pred, target)
        self.spearman_corrcoef(pred, target)
        self.smape(pred, target)
        self.wmape(pred, target)

    def log_all_test_metrics(self):

        # self.log('test/concordance_cc', self.concordance_cc, on_epoch=True, on_step=False, prog_bar=True)
        # self.log('test/cosine_similarity', self.cosine_similarity,
        #  on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/explained_variance', self.explained_variance,
                 on_epoch=True, on_step=False, prog_bar=True)
        # self.log('test/kendall_tau', self.kendall_tau, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/log_cosh_error', self.log_cosh_error, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/mae', self.mae, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/mape', self.mape, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/mse', self.mse, on_epoch=True, on_step=False, prog_bar=True)
        # self.log('test/msle', self.msle, on_epoch=True, on_step=False, prog_bar=True)
        # self.log('test/pearson_corrcoef', self.pearson_corrcoef, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/r2score', self.r2score, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/spearman_corrcoef', self.spearman_corrcoef,
                 on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/smape', self.smape, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/wmape', self.wmape, on_epoch=True, on_step=False, prog_bar=True)

    def calculate_val_metrics(self, pred, target):
        m_pred = (pred * (self.max_c - self.min_c)) + self.min_c
        m_target = (target * (self.max_c - self.min_c)) + self.min_c

        self.val_mape(m_pred, m_target)
        self.val_mse(m_pred, m_target)
        self.val_mae(m_pred, m_target)

    def log_all_val_metrics(self):
        self.log('val/mape', self.val_mape, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/mae', self.val_mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/mse', self.val_mse, on_step=False, on_epoch=True, prog_bar=False)

    def set_max_min(self, max_val, min_val):
        self.max_c = max_val
        self.min_c = min_val

    @staticmethod
    def get_default_config():
        return {}

    @classmethod
    def fromOptunaTrial(cls, trial):
        return cls()
