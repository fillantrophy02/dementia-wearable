import numpy as np
import torch
import torchmetrics
from components.experiment_recorder import log_model_metric
from components.model import SleepPatchTST
from components.data_loader import val_dataloaders
from components.metrics import Metrics, plot_and_save_auc_curve
from config import *

def eval_model(model, epoch=num_epochs-1, fold=k_folds-1):
    model.eval()
    metrics = Metrics(['auc', 'f1_score', 'cm'], prefix=f'val_{fold}_')

    with torch.no_grad():
        labels, preds, = [], []
        losses_per_batch = []

        for batch in val_dataloaders[fold]:
            inputs_batch, outputs_batch = batch
            inputs_re = inputs_batch.to(device)
            outputs_re = outputs_batch.to(device)

            pred = model(inputs_re)
            loss = model.loss(pred.float(), outputs_re.float())
            labels.extend(outputs_re.cpu().numpy())
            preds.extend(pred.cpu().numpy())

            # Compute test metrics
            losses_per_batch.append(loss.item())
            metrics.update(pred, outputs_re)

        avg_loss = sum(losses_per_batch) / len(losses_per_batch)
        metrics.report()
        log_model_metric(f'val_{fold}_loss', avg_loss, epoch)
        metrics.log_to_experiment_tracker(epoch)
        # plot_and_save_auc_curve("visualizations/roc.png", np.array(labels), np.array(preds))
        results = metrics.compute()
    return results

if __name__ == '__main__':
    model = SleepPatchTST(input_size=input_size).to(device)
    avg_results = {}

    for fold in range(k_folds):
        model.load_state_dict(torch.load(f"ckpts/model_{fold}.pth"))
        results = eval_model(model, fold=fold)

        for key, value in results.items():
            if key not in avg_results:
                avg_results[key] = []
            avg_results[key].append(value)

    print(f"Average result across {k_folds} folds:")
    for key in avg_results:
        avg_results[key] = sum(avg_results[key]) / k_folds
        print(f"{key}: {value:.4f}")
