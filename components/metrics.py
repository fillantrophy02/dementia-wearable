import sys
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay
import torchmetrics
from config import *
from components.experiment_recorder import log_model_metric

class Metrics:
    CM_METRIC_NAMES = ["sensitivity", "specificity", "precision", "accuracy"]

    def __init__(self, metric_names: list[str] = [], prefix=''):
        self.prefix = prefix
        self._instantiate_metrics(metric_names)

    def _instantiate_metric(self, name):
        if name == 'auc':
            return torchmetrics.AUROC(task='binary').to(device)
        elif name == 'cm':
            return torchmetrics.ConfusionMatrix(task='binary').to(device)
        elif name == 'f1_score':
            return torchmetrics.F1Score(task='binary').to(device)

    def _instantiate_metrics(self, metric_names):
        self.metrics = {}
        for name in metric_names:
            self.metrics[f"{self.prefix}{name}"] = self._instantiate_metric(name)

    def update(self, pred, output):
        for name in self.metrics:
            self.metrics[name](pred, output)
        
    def compute(self):
        results = {}
        for name in self.metrics:
            if 'cm' not in name:
                results[name] = self.metrics[name].compute().item()
            else:
                cm = self.metrics[name].compute().tolist()
                cm_results = self._calculate_cm_metrics(cm)
                for i in range(4):
                    results[f"{self.prefix}{Metrics.CM_METRIC_NAMES[i]}"] = cm_results[i]
        return results

    def reset(self):
        for name in self.metrics:
            self.metrics[name].reset()

    def log_to_experiment_tracker(self, epoch):
        results = self.compute()
        for name in results:
            log_model_metric(name, results[name], epoch) 
        
    def report(self):
        results = self.compute()
        for name in results:
            print(f"{name}: {results[name]:.4f}", end = '    ')

    def _calculate_cm_metrics(self, cm):
        TN, FP = cm[0]
        FN, TP = cm[1]

        safe_divide = lambda num, denom: num / denom if denom != 0 else 0.0

        sensitivity = safe_divide(TP, TP + FN)
        specificity = safe_divide(TN, TN + FP)
        precision = safe_divide(TP, TP + FP)
        accuracy = safe_divide(TP + TN, TP + TN + FP + FN)

        return sensitivity, specificity, precision, accuracy

    def calculate_tpr_fpr(self, cm):
        tn, fp = cm[0]
        fn, tp = cm[1]
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        return tpr, fpr

def plot_and_save_auc_curve(filepath, y_true, y_pred):
    display = RocCurveDisplay.from_predictions(y_true, y_pred)
    display.plot()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved ROC curve to {filepath}.")

def get_metric_fold_name(fold=None):
    metric_fold_name = metric_to_choose_best_model.split('_')
    metric_fold_name = f'{metric_fold_name[0]}_{fold}_{metric_fold_name[1]}'
    return metric_fold_name