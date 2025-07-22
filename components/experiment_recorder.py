import os
import mlflow
import mlflow.pytorch
import config
from config import *

debug_mode = False
if not debug_mode: # TODO fix this thing
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.start_run()

    # Log config.py variables aka hyperparams
    config_vars = {
        key: getattr(config, key)
        for key in dir(config)
        if not key.startswith('__')
    }
    mlflow.log_params(config_vars)

def log_model_artifacts(model, fold=k_folds-1):
    if not debug_mode:
        with open("model_summary.txt", "w") as f:
            f.write(str(model))
        mlflow.log_artifact("model_summary.txt")
        os.remove("model_summary.txt")
        mlflow.log_artifact(f"ckpts/{dataset}/{chosen_model}_{fold}{f'_TL_{transfer_learning_dataset}' if is_transfer_learning else ''}.pth")
        mlflow.pytorch.log_model(model, "model")

def log_model_metric(name, value, epoch):
    if not debug_mode:
        mlflow.log_metric(name, value, epoch)
