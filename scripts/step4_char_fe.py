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
import joblib

# --- Configuration ---
DATA_FILE = 'data/dl_data.npz'
MODEL_PATH = 'outputs/siamese_char_cnn.pt'
RESULTS_FILE = 'results_log.csv'
SUBMISSION_XGB = 'outputs/submission_char_fe_xgb.npy'
SUBMISSION_RIDGE = 'outputs/submission_char_fe_ridge.npy'
BATCH_SIZE = 128
EMBEDDING_DIM = 64
HIDDEN_DIM = 256

# --- Device Config ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            # 1. Absolute Difference
            diff = np.abs(h1 - h2)
            # 2. Element-wise Product
            prod = h1 * h2
            # 3. Cosine Similarity
            cosine = np.sum(h1 * h2, axis=1, keepdims=True) / (
                np.linalg.norm(h1, axis=1, keepdims=True) * np.linalg.norm(h2, axis=1, keepdims=True) + 1e-8
            )
            # 4. Euclidean Distance
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
    header = not os.path.exists(RESULTS_FILE)
    res_df.to_csv(RESULTS_FILE, mode='a', header=header, index=False)

def main():
    print(f"Using device: {device}")
    
    # 1. Load Data
    X_s_tr, X_d_tr, y_train, X_s_val, X_d_val, y_val, X_s_te, X_d_te, vocab_size = load_data()
    
    # 2. Load Siamese Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run train_model.py first.")
        
    print(f"Loading Siamese model from {MODEL_PATH}...")
    # Load state dict but filter out FC layers since we changed init args potentially, 
    # or just use strict=False if architecture is identical. 
    # Actually, the SiameseCNN class here matches the training one, so strict load works.
    siamese = SiameseCNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    
    # Note: We need to load state_dict carefully. 
    # The saved model has the full architecture including the final FC layers.
    # Our feature extractor definition here only has 'encoder'.
    # So we load the state dict and only keep keys starting with 'encoder'.
    state_dict = torch.load(MODEL_PATH, map_location=device)
    encoder_state_dict = {k: v for k, v in state_dict.items() if k.startswith('encoder')}
    siamese.load_state_dict(encoder_state_dict)
    
    # 3. Extract Features
    print("Extracting features (this may take a while)...")
    
    train_loader = DataLoader(CharDataset(X_s_tr, X_d_tr), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(CharDataset(X_s_val, X_d_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(CharDataset(X_s_te, X_d_te), batch_size=BATCH_SIZE, shuffle=False)
    
    X_train_feats = extract_features(siamese, train_loader, device)
    print(f"Train features shape: {X_train_feats.shape}")
    
    X_val_feats = extract_features(siamese, val_loader, device)
    print(f"Val features shape: {X_val_feats.shape}")
    
    X_test_feats = extract_features(siamese, test_loader, device)
    print(f"Test features shape: {X_test_feats.shape}")
    
    # 4. Train Model 1: XGBoost on Char Features
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
    te_pred = np.clip(te_pred, 1.0, 3.0)
    np.save(SUBMISSION_XGB, te_pred)
    
    # 5. Train Model 2: Ridge on Char Features
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
    te_pred = np.clip(te_pred, 1.0, 3.0)
    np.save(SUBMISSION_RIDGE, te_pred)
    
    print("\nDone with Char Feature Extraction!")

if __name__ == "__main__":
    main()
