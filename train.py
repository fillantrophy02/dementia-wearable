import copy
import sys
import torch
from components.early_stopper import EarlyStopper
from components.model import SleepLSTM, SleepPatchTST
from components.data_loader import train_dataloaders
from components.metrics import Metrics, get_metric_fold_name
from config import *
from eval import eval_across_kfolds, eval_model
from components.experiment_recorder import log_model_artifacts, log_model_metric

def train_model(model, fold=k_folds-1): # default last k-fold split
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = Metrics(['auc', 'f1_score', 'cm'], prefix=f'train_{fold}_')
    best_score, best_state, best_epoch = -9999999, None, None
    metric_fold_name = get_metric_fold_name(fold=fold)
    
    for epoch in range(num_epochs):
        model.train() 
        losses_per_batch = []

        for batch in train_dataloaders[fold]:
            optimizer.zero_grad()

            inputs_batch, outputs_batch = batch
            inputs_re = inputs_batch.to(device)
            outputs_re = outputs_batch.to(device)
            
            # Forward pass
            pred = model(inputs_re)
            loss = model.loss(pred.float(), outputs_re.float())

            # Update metrics
            metrics.update(pred, outputs_re)
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            losses_per_batch.append(loss.item())

        avg_loss = sum(losses_per_batch) / len(losses_per_batch)
        print(f'\nEpoch [{epoch+1}/{num_epochs}]  ', f'loss: {avg_loss:.4f}', end='    ')
        metrics.report()
        
        log_model_metric(f'train_loss_{fold}', avg_loss, epoch)
        metrics.log_to_experiment_tracker(epoch)
        
        metrics.reset()

        if (epoch+1) % 1 == 0:
            results, _ = eval_model(model, epoch=epoch, fold=fold)
            
            if results[metric_fold_name] > best_score:
                best_score, best_state, best_epoch = results[metric_fold_name], copy.deepcopy(model.state_dict()), epoch

    print(f"\nEpoch {best_epoch} had the highest {metric_to_choose_best_model} at {best_score:.4f}. Saving model....")
    torch.save(best_state, f"ckpts/{chosen_model}_{fold}{special_mode_suffix}.pth")
    log_model_artifacts(model, fold=fold)
    print("Model saved.")

if __name__ == '__main__':  
    if chosen_model == "PatchTST":
        model = SleepPatchTST(input_size=input_size)
    elif chosen_model == "LSTM":
        model = SleepLSTM(input_size=input_size)
    else:
        raise Exception(f"Model_{chosen_model} doesn't exist. Please select another value for 'chosen_moden' in 'config.py'.")
    model = model.to(device)

    for fold in range(k_folds):
        train_model(model, fold=fold)
    eval_across_kfolds()