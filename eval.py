import numpy as np
import torch
import torchmetrics
from components.experiment_recorder import log_model_metric
from components.model import SleepPatchTST
from components.data_loader import test_dataloader, input_size
from components.metrics import Metrics, plot_and_save_auc_curve
from config import device, num_epochs

def eval_model(model, epoch=num_epochs-1):
    model.eval()
    metrics = Metrics(['auc', 'f1_score', 'cm'], prefix='val_')

    with torch.no_grad():
        labels, preds, = [], []
        losses_per_batch = []

        for batch in test_dataloader:
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
        log_model_metric('val_loss', avg_loss, epoch)
        metrics.log_to_experiment_tracker(epoch)
        # plot_and_save_auc_curve("visualizations/roc.png", np.array(labels), np.array(preds))
        results = metrics.compute()
    return results

if __name__ == '__main__':
    model = SleepPatchTST(input_size=input_size).to(device)
    model.load_state_dict(torch.load("ckpts/model.pth"))
    eval_model(model)