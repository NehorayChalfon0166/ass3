import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# --- Configuration ---
DATA_FILE = 'data/dl_data.npz'
RESULTS_FILE = 'results_log.csv'
MODEL_SAVE_PATH = 'outputs/siamese_char_cnn.pt'
SUBMISSION_FILE = 'outputs/submission_siamese_char.npy'
PLOT_FILE = 'outputs/training_history_char_siamese.png'
BATCH_SIZE = 64 
EPOCHS = 15
EMBEDDING_DIM = 64
HIDDEN_DIM = 256
LEARNING_RATE = 0.001

class CharDataset(Dataset):
    def __init__(self, search, desc, labels=None):
        # We ensure they are the correct length and type for PyTorch
        self.search = torch.tensor(search.astype(np.int64))
        self.desc = torch.tensor(desc.astype(np.int64))
        if labels is not None:
            self.labels = torch.tensor(labels.astype(np.float32))
        else:
            self.labels = None
        
    def __len__(self):
        return len(self.search)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.search[idx], self.desc[idx], self.labels[idx]
        return self.search[idx], self.desc[idx]

class CharCNNEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super(CharCNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        self.convs = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) 
        )
        self.fc = nn.Linear(256, hidden_dim)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2) 
        x = self.convs(x).squeeze(-1) 
        x = self.fc(x)
        return x

class SiameseCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, target_mean=2.38):
        super(SiameseCNN, self).__init__()
        self.encoder = CharCNNEncoder(vocab_size, emb_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        with torch.no_grad():
            self.fc[-1].bias.fill_(target_mean)

    def forward(self, s, d):
        h1 = self.encoder(s)
        h2 = self.encoder(d)
        
        diff = torch.abs(h1 - h2)
        prod = h1 * h2
        
        combined = torch.cat([h1, h2, diff, prod], dim=1)
        return self.fc(combined).squeeze(-1)

def load_data():
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Please run step1_preprocess.py first.")
        
    with np.load(DATA_FILE, allow_pickle=True) as data:
        char_to_int = data['char_to_int'].item()
        vocab_size = len(char_to_int) + 1
        return (data['X_train_search'], data['X_train_desc'], data['y_train'],
                data['X_val_search'], data['X_val_desc'], data['y_val'],
                data['X_test_search'], data['X_test_desc'], vocab_size)

def log_results(runtime, train_rmse, val_rmse, train_mae, val_mae):
    # Logs TRAIN/VAL results only.
    res_df = pd.DataFrame([{
        'Model type': 'Character level CNN (Siamese)',
        'runtime': f"{runtime:.2f} sec",
        'Train RMSE': f"{train_rmse:.4f}",
        'Val-RMSE': f"{val_rmse:.4f}",
        'Test-RMSE': "N/A (See evaluate.py)",
        'Train MAE': f"{train_mae:.4f}",
        'Val-MAE': f"{val_mae:.4f}",
        'Test-MAE': "N/A (See evaluate.py)"
    }])
    res_df.to_csv(RESULTS_FILE, mode='a', header=not os.path.exists(RESULTS_FILE), index=False)
    print(f"\nTraining Results saved to {RESULTS_FILE}")

def main():
    # ... (existing device and loading code) ...
    # ...
    # After training loop:
    # Load best model for final Train/Val/Test evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    model.eval()
    def get_metrics(loader):
        preds, targets = [], []
        with torch.no_grad():
            for s, d, y in loader:
                s, d = s.to(device), d.to(device)
                out = model(s, d)
                preds.extend(out.cpu().numpy())
                targets.extend(y.numpy())
        preds = np.clip(preds, 1.0, 3.0)
        rmse = np.sqrt(np.mean((np.array(targets) - np.array(preds))**2))
        mae = np.mean(np.abs(np.array(targets) - np.array(preds)))
        return rmse, mae

    tr_rmse, tr_mae = get_metrics(DataLoader(CharDataset(X_s_tr, X_d_tr, y_tr), batch_size=BATCH_SIZE))
    val_rmse, val_mae = get_metrics(DataLoader(CharDataset(X_s_val, X_d_val, y_val), batch_size=BATCH_SIZE))

    # Test Prediction
    test_loader = DataLoader(CharDataset(X_s_te, X_d_te, labels=None), batch_size=BATCH_SIZE)
    # ... (rest of the test pred logic) ...
    
    # Log Results
    log_results(total_time, tr_rmse, val_rmse, tr_mae, val_mae)


if __name__ == "__main__":
    main()