import re
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torchmetrics
from components.experiment_recorder import log_model_metric
from models.GRU import GRU
from models.PatchTST import PatchTST
from models.LSTM import LSTM
from config import *
from components.metrics import Metrics, plot_and_save_auc_curve
from models.Time_Series_Transformer import TimeSeriesTransformer
from models.Vanilla_Transformer import VanillaTransformer

def eval_model(model, epoch=num_epochs-1, fold=None, log_to_experiment_tracker=True):
    model.eval()
    metrics = Metrics(['auc', 'f1_score', 'cm'], prefix=f'val{f'_{fold}' if fold is not None else ''}_')

    if dataset == 'fitbit_mci':
        from data.fitbit_mci.data_loader import val_dataloaders, val_ids, val_num_days, val_labels
        num_positive_days, day_idx = {pid: 0 for pid in val_num_days[fold].keys()}, 0
        dataloader = val_dataloaders[fold]
    elif dataset == 'wearable_korean':
        from data.wearable_korean.data_loader import test_dataloader
        dataloader = test_dataloader

    with torch.no_grad():
        labels, preds, = [], []
        losses_per_batch = []

        for batch in dataloader:
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

            if dataset == 'fitbit_mci':
                # Re-calculate number of days classified as MCI or not
                pred_lst = pred.cpu().numpy()
                for i in range(len(pred_lst)):
                    pid = val_ids[fold][day_idx]
                    num_positive_days[pid] += int(pred_lst[i][0] >= 0.5)
                    day_idx += 1

        avg_loss = sum(losses_per_batch) / len(losses_per_batch)
        print(f'val{f'_{fold}' if fold is not None else ''}_loss: {avg_loss:.4f}', end='    ')
        metrics.report()

        if log_to_experiment_tracker:
            log_model_metric(f'val{f'_{fold}' if fold is not None else ''}_loss', avg_loss, epoch)
            metrics.log_to_experiment_tracker(epoch)

        results = metrics.compute()
        results[f'val{f'_{fold}' if fold is not None else ''}_loss'] = -avg_loss

        if dataset == 'fitbit_mci':
            # Calculate fraction of days labelled as MCI or not, then take AUC
            predicted_probs = {pid: num_positive_days[pid] / val_num_days[fold][pid] for pid in sorted(num_positive_days)}
            predicted_labels = {pid: 1 if prob >= 0.5 else 0 for pid, prob in predicted_probs.items()}
            majority_vote_auc = roc_auc_score(list(val_labels[fold].values()), list(predicted_labels.values()))
            results[f'val_{fold}_majority_vote_auc'] = majority_vote_auc

    return results

def eval_across_kfolds(model):
    avg_results = {}
    
    for fold in range(k_folds):
        model.load_state_dict(torch.load(f"ckpts/{dataset}/{chosen_model}_{fold}{f'_TL_{transfer_learning_dataset}' if is_transfer_learning else ''}.pth"))
        results = eval_model(model, fold=fold, log_to_experiment_tracker=False)

        for key, value in results.items():
            base_key = re.sub(r'_\d+(?=_)', '', key)
            if base_key not in avg_results:
                avg_results[base_key] = []
            avg_results[base_key].append(value)

    print(f"\n\nAverage result across {k_folds} folds:")
    for key in avg_results:
        avg_results[key] = np.nanmean(avg_results[key])
        print(f"{key}: {avg_results[key]:.4f}")

if __name__ == '__main__':
    if chosen_model == "PatchTST":
        model = PatchTST()
    elif chosen_model == "LSTM":
        model = LSTM()
    elif chosen_model == "GRU":
        model = GRU()
    elif chosen_model == "TimeSeriesTransformer":
        model = TimeSeriesTransformer()
    elif chosen_model == "VanillaTransformer":
        model = VanillaTransformer()
    else:
        raise Exception(f"Model_{chosen_model} doesn't exist. Please select another value for 'chosen_moden' in 'config.py'.")
    
    model = model.to(device)

    if dataset == 'fitbit_mci':
        eval_across_kfolds(model=model)
    elif dataset == 'wearable_korean':
        eval_model(model=model)