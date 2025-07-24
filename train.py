import copy
import sys
import torch
from models.GRU import GRU
from models.Time_Series_Transformer import TimeSeriesTransformer
from models.Vanilla_Transformer import VanillaTransformer
from models.LSTM import LSTM
from models.PatchTST import PatchTST
from components.metrics import Metrics, get_metric_fold_name
from config import *
from eval import eval_across_kfolds, eval_model
from components.experiment_recorder import log_model_artifacts, log_model_metric

def freeze_backbone(model, freeze=True):
    for param in model.backbone.parameters():
        param.requires_grad = not freeze

def train_model(model, fold=None): # default last k-fold split
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = Metrics(['auc', 'f1_score', 'cm'], prefix=f'train{f'_{fold}' if fold is not None else ''}_')
    freeze_threshold_epoch = int(num_epochs * freeze_threshold)
    best_score, best_state, best_epoch = -9999999, None, None

    if dataset == 'fitbit_mci':
        from data.fitbit_mci.data_loader import train_dataloaders
        main_metric_name = get_metric_fold_name(fold=fold)
        dataloader = train_dataloaders[fold]
    elif dataset == 'wearable_korean':
        from data.wearable_korean.data_loader import train_dataloader
        main_metric_name = metric_to_choose_best_model
        dataloader = train_dataloader
    
    for epoch in range(num_epochs):
        model.train() 
        losses_per_batch = []

        if is_transfer_learning:
            freeze_backbone(model, freeze=True)
            if epoch == freeze_threshold_epoch:
                freeze_backbone(model, freeze=False)
        
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=0.001)

        for batch in dataloader:
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
        
        log_model_metric(f'train_loss{f'_{fold}' if fold is not None else ''}', avg_loss, epoch)
        metrics.log_to_experiment_tracker(epoch)
        metrics.reset()

        if (epoch + 1) % 1 == 0:
            results = eval_model(model, epoch=epoch, fold=fold)
            if results[main_metric_name] > best_score:
                best_score, best_state, best_epoch = results[main_metric_name], copy.deepcopy(model.state_dict()), epoch

    print(f"\nEpoch {best_epoch} had the highest {metric_to_choose_best_model} at {best_score:.4f}. Saving model....")
    model_name = f"{chosen_model}{f'_{fold}' if fold is not None else ''}{f'_TL_{transfer_learning_dataset}' if is_transfer_learning else ''}.pth"
    torch.save(best_state, f"ckpts/{dataset}/{model_name}")
    log_model_artifacts(model, fold=fold)
    print(f"Model saved as {model_name}")

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

    if dataset == 'fitbit_mci': # Perform k-fold cross validation
        for fold in range(k_folds):
            train_model(model, fold=fold)
        eval_across_kfolds(model=model)
    elif dataset == 'wearable_korean': # Perform normal validation
        train_model(model)