import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchmetrics
from neptune.types import File
from sklearn.metrics import ConfusionMatrixDisplay


class LitBaseCls(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.model_name = 'BaseCls'

        self.tm_val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

        self.tm_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.tm_f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.tm_precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.tm_recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.tm_confusion_matrix = torchmetrics.ConfusionMatrix(
            task='multiclass', num_classes=num_classes)

    def calculate_test_metrics(self, pred, target):
        self.tm_accuracy(pred, target)
        self.tm_f1_score(pred, target)
        self.tm_recall(pred, target)
        self.tm_precision(pred, target)
        self.tm_confusion_matrix(pred, target)

    def log_all_test_metrics(self):
        disp = ConfusionMatrixDisplay(self.tm_confusion_matrix.compute().cpu().numpy())
        disp.plot(cmap=plt.cm.get_cmap("Blues"), xticks_rotation='vertical')
        self.logger.experiment["test/confusion_matrix"].upload(File.as_image(disp.figure_))

        self.log('test/accuracy', self.tm_accuracy, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/f1_score', self.tm_f1_score, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/recall', self.tm_recall, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test/precision', self.tm_precision, on_epoch=True, on_step=False, prog_bar=True)

    def calculate_val_metrics(self, pred, target):
        self.tm_val_accuracy(pred, target)

    def log_all_val_metrics(self):
        self.log('val/accuracy', self.tm_val_accuracy, on_step=False, on_epoch=True, prog_bar=False)

    @staticmethod
    def get_default_config():
        return {}

    @classmethod
    def fromOptunaTrial(cls, trial):
        return cls()
