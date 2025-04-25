import copy
import sys
import torch
from components.early_stopper import EarlyStopper
from components.model import SleepPatchTST
from components.data_loader import train_dataloader
from components.metrics import Metrics
from config import *
from eval import eval_model
from components.experiment_recorder import log_model_artifacts, log_model_metric

def train_model(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = Metrics(['auc', 'f1_score', 'cm'])
    best_score, best_state, best_epoch = 0, None, None
    
    for epoch in range(num_epochs):
        model.train() 
        losses_per_batch = []

        for batch in train_dataloader:
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
        
        log_model_metric('loss', avg_loss, epoch)
        metrics.log_to_experiment_tracker(epoch)
        
        metrics.reset()

        if (epoch+1) % 1 == 0:
            results = eval_model(model, epoch=epoch)
            if results[metric_to_choose_best_model] > best_score:
                best_score, best_state, best_epoch = results[metric_to_choose_best_model], copy.deepcopy(model.state_dict()), epoch

    print(f"\nEpoch {best_epoch} had the highest {metric_to_choose_best_model} at {best_score:.4f}. Saving model....")
    torch.save(best_state, "ckpts/model.pth")
    log_model_artifacts(model)
    print("Model saved.")

if __name__ == '__main__':  
    model = SleepPatchTST(input_size=input_size).to(device)
    train_model(model)