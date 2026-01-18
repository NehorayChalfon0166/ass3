import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import time

# --- Configuration ---
DATA_FILE = 'data/dl_data.npz'
MODEL_PATH = 'outputs/siamese_char_cnn.pt'
RESULTS_FILE = 'results_log.csv'
SUBMISSION_XGB = 'outputs/submission_char_fe_xgb.npy'
SUBMISSION_RIDGE = 'outputs/submission_char_fe_ridge.npy'
BATCH_SIZE = 128
EMBEDDING_DIM = 64
HIDDEN_DIM = 256

# --- Re-define Model Classes (Must match training) ---
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
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super(SiameseCNN, self).__init__()
        self.encoder = CharCNNEncoder(vocab_size, emb_dim, hidden_dim)
        # We don't need the rest for feature extraction

    def forward(self, s, d):
        return self.encoder(s), self.encoder(d)

class CharDataset(Dataset):
    def __init__(self, search, desc):
        self.search = torch.tensor(search.astype(np.int64))
        self.desc = torch.tensor(desc.astype(np.int64))
    def __len__(self): return len(self.search)
    def __getitem__(self, idx): return self.search[idx], self.desc[idx]

def extract_features(model, loader, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for s, d in loader:
            s, d = s.to(device), d.to(device)
            h1, h2 = model(s, d)
            
            h1 = h1.cpu().numpy()
            h2 = h2.cpu().numpy()
            
            # Create Interaction Features
            diff = np.abs(h1 - h2)
            prod = h1 * h2
            cosine = np.sum(h1 * h2, axis=1, keepdims=True) / (
                np.linalg.norm(h1, axis=1, keepdims=True) * np.linalg.norm(h2, axis=1, keepdims=True) + 1e-8
            )
            euclid = np.linalg.norm(h1 - h2, axis=1, keepdims=True)
            
            # Concatenate all
            batch_feats = np.hstack([h1, h2, diff, prod, cosine, euclid])
            feats.append(batch_feats)
            
    return np.vstack(feats)

def log_results(model_name, runtime, train_rmse, val_rmse, train_mae, val_mae):
    print(f"Logging {model_name}...")
    res_df = pd.DataFrame([{
        'Model type': model_name,
        'runtime': f"{runtime:.2f} sec",
        'Train RMSE': f"{train_rmse:.4f}",
        'Val-RMSE': f"{val_rmse:.4f}",
        'Test-RMSE': "N/A (See evaluate.py)",
        'Train MAE': f"{train_mae:.4f}",
        'Val-MAE': f"{val_mae:.4f}",
        'Test-MAE': "N/A (See evaluate.py)"
    }])
    res_df.to_csv(RESULTS_FILE, mode='a', header=not os.path.exists(RESULTS_FILE), index=False)

def main():
    # ... (existing loading and feature extraction code) ...

    # 4. Train Model 1: XGBoost on Char Features...
    print("\nTraining Model 1: XGBoost on Char Features...")
    start = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        n_jobs=-1,
        random_state=42
    )
    xgb_model.fit(X_train_feats, y_train)
    rt = time.time() - start
    
    # Eval XGB
    tr_pred = np.clip(xgb_model.predict(X_train_feats), 1.0, 3.0)
    val_pred = np.clip(xgb_model.predict(X_val_feats), 1.0, 3.0)
    tr_rmse = np.sqrt(mean_squared_error(y_train, tr_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    tr_mae = mean_absolute_error(y_train, tr_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    print(f"XGB Results - Train RMSE: {tr_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
    log_results("FE (Char) + XGBoost", rt, tr_rmse, val_rmse, tr_mae, val_mae)
    
    # Save XGB Preds
    te_pred = xgb_model.predict(X_test_feats)
    np.save(SUBMISSION_XGB, te_pred)
    
    # 5. Train Model 2: Ridge on Char Features...
    print("\nTraining Model 2: Ridge on Char Features...")
    start = time.time()
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_feats, y_train)
    rt = time.time() - start
    
    # Eval Ridge
    tr_pred = np.clip(ridge_model.predict(X_train_feats), 1.0, 3.0)
    val_pred = np.clip(ridge_model.predict(X_val_feats), 1.0, 3.0)
    tr_rmse = np.sqrt(mean_squared_error(y_train, tr_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    tr_mae = mean_absolute_error(y_train, tr_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    print(f"Ridge Results - Train RMSE: {tr_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
    log_results("FE (Char) + Ridge", rt, tr_rmse, val_rmse, tr_mae, val_mae)

    
    # Save Ridge Preds
    te_pred = ridge_model.predict(X_test_feats)
    np.save(SUBMISSION_RIDGE, te_pred)
    
    print("\nDone with Char Feature Extraction!")

if __name__ == "__main__":
    main()