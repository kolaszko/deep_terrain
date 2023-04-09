import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchmetrics
from neptune.types import File
from sklearn.metrics import ConfusionMatrixDisplay


class LitBaseCls(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.model_name = 'BaseCls'

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)
        self.precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes)
        self.recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes)

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
