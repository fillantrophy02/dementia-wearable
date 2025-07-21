import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torchmetrics
from components.experiment_recorder import log_model_metric
from components.model import SleepLSTM, SleepPatchTST
from components.data_loader import val_dataloaders, val_ids, val_num_days, val_labels
from components.metrics import Metrics, plot_and_save_auc_curve
from config import *

def eval_model(model, epoch=num_epochs-1, fold=k_folds-1, log_to_experiment_tracker=True):
    model.eval()
    metrics = Metrics(['auc', 'f1_score', 'cm'], prefix=f'val_{fold}_')

    with torch.no_grad():
        labels, preds, = [], []
        losses_per_batch = []
        num_positive_days = {pid: 0 for pid in val_num_days[fold].keys()}
        day_idx = 0

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

            # Re-calculate number of days classified as MCI or not
            pred_lst = pred.cpu().numpy()
            for i in range(len(pred_lst)):
                pid = val_ids[fold][day_idx]
                num_positive_days[pid] += int(pred_lst[i][0] >= 0.5)
                day_idx += 1

        avg_loss = sum(losses_per_batch) / len(losses_per_batch)
        print(f'val_{fold}_loss: {avg_loss:.4f}', end='    ')
        metrics.report()

        if log_to_experiment_tracker:
            log_model_metric(f'val_{fold}_loss', avg_loss, epoch)
            metrics.log_to_experiment_tracker(epoch)

        # plot_and_save_auc_curve("visualizations/roc.png", np.array(labels), np.array(preds))
        results = metrics.compute()
        results[f'val_{fold}_loss'] = -avg_loss

        # Calculate fraction of days labelled as MCI or not, then take AUC
        predicted_probs = {pid: num_positive_days[pid] / val_num_days[fold][pid] for pid in sorted(num_positive_days)}
        predicted_labels = {pid: 1 if prob >= 0.5 else 0 for pid, prob in predicted_probs.items()}
        majority_vote_auc = roc_auc_score(list(val_labels[fold].values()), list(predicted_labels.values()))

    return results, majority_vote_auc

def eval_across_kfolds():
    avg_results, majority_vote_aucs = {}, []
    for fold in range(k_folds):
        if chosen_model == "PatchTST":
            model = SleepPatchTST(input_size=input_size)
        elif chosen_model == "LSTM":
            model = SleepLSTM(input_size=input_size)

        model = model.to(device)
        model.load_state_dict(torch.load(f"ckpts/{chosen_model}_{fold}{special_mode_suffix}.pth"))
        results, majority_vote_auc = eval_model(model, fold=fold, log_to_experiment_tracker=False)

        for key, value in results.items():
            base_key = f"{key.split('_')[0]}_{key.split('_')[-1]}"
            if base_key not in avg_results:
                avg_results[base_key] = []
            avg_results[base_key].append(value)

        majority_vote_aucs.append(majority_vote_auc)

    print(f"\n\nAverage result across {k_folds} folds:")
    print("majority_vote_val_auc:", np.nanmean(majority_vote_aucs))
    for key in avg_results:
        avg_results[key] = sum(avg_results[key]) / k_folds
        print(f"{key}: {avg_results[key]:.4f}")

if __name__ == '__main__':
    eval_across_kfolds()