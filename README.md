# Medical Time Series Classification

This is the code base for experimenting with **time-series physiological signals** classification tasks with respect to **dementia/MCI patients**. There are **5 models** implemented: PatchTST, LSTM, GRU, VanillaTransformer, and TimeSeriesTransformer.

## 📂 Datasets

This repo includes 2 publicly available datasets. The first is the Korean dataset used by [Hong et al.](https://www.mdpi.com/2227-7390/12/20/3208), which includes data from 174 participants equipped with ring-shaped wearable devices - 111 cognitively normal (CN) and 63 with MCI or dementia - captured over 2.5 months across 32 sleep-related features. The second is the Fitbit dataset from [Xu et al.](https://doi.org/10.2196/55575), comprising 20 participants (8 CN and 12 MCI) over ~1 month with features covering 18 activity signals, 17 heart rate, and 18 sleep. Sleep features are further sub-divided into 6 main sleep and 12 nap features.


## 🔧 Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the environment and install dependencies:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Start the MLflow tracking server in a separate terminal:
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```

4. Configure the model in `config.py`:
   ```python
   dataset = "wearable_korean"
   chosen_model = "PatchTST" 
   is_transfer_learning = True
   ...
   ```

## 🏋️‍♀️ Training

To train the selected model:
```bash
python train.py
```

## ✅ Testing

To evaluate the trained model on the validation set:
```bash
python eval.py
```

## 📓 Checkpoints

The following model checkpoints are saved in `ckpts/{dataset}`:

| Dataset | Name | Model | Data group | Is transfer learning? | Val AUC |
| - | - | - | - | - | - |
| Fitbit | `PatchTST_N.pth` where `N = 0..4`| PatchTST | Activities + Heart rate + Main sleep | N | TODO |
| Fitbit | `PatchTST_N_TL_wearable_korean.pth` where `N = 0..4`| PatchTST | Korean-Fitbit Common Features | Y | TODO |
| Korean | `PatchTST.pth` | PatchTST | Sleep | N | 0.94 |
| Korean | `PatchTST_TL_fitbit_mci.pth` | PatchTST | Korean-Fitbit Common Features | Y | 0.92 |