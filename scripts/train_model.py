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

# --- Device Config ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CharDataset(Dataset):
    def __init__(self, search, desc, labels=None):
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
        # x: (batch, seq_len) -> (batch, emb, seq_len)
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
        # Initialize bias to mean target value for faster convergence
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
    header = not os.path.exists(RESULTS_FILE)
    res_df.to_csv(RESULTS_FILE, mode='a', header=header, index=False)
    print(f"\nTraining Results saved to {RESULTS_FILE}")

def main():
    print(f"Using device: {device}")
    
    # 1. Load Data
    X_s_tr, X_d_tr, y_tr, X_s_val, X_d_val, y_val, X_s_te, X_d_te, vocab_size = load_data()
    
    # 2. Setup
    model = SiameseCNN(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(CharDataset(X_s_tr, X_d_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CharDataset(X_s_val, X_d_val, y_val), batch_size=BATCH_SIZE)
    
    print(f"Model initialized. Vocab size: {vocab_size}")
    print("Starting training...")
    
    # 3. Training Loop
    start_time = time.time()
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for s, d, y in train_loader:
            s, d, y = s.to(device), d.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(s, d)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * s.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s, d, y in val_loader:
                s, d, y = s.to(device), d.to(device), y.to(device)
                output = model(s, d)
                loss = criterion(output, y)
                val_loss += loss.item() * s.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")
    print(f"Best model saved to {MODEL_SAVE_PATH}")
    
    # 4. Plot History
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Siamese Network Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(PLOT_FILE)
    print(f"Plot saved to {PLOT_FILE}")
    
    # 5. Final Evaluation (Best Model)
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH)) # ensure we use the best one
    model.eval()
    
    def get_metrics(loader):
        preds, targets = [], []
        with torch.no_grad():
            for s, d, y in loader:
                s, d, y = s.to(device), d.to(device), y.to(device)
                out = model(s, d)
                preds.extend(out.cpu().numpy())
                targets.extend(y.numpy())
        
        preds = np.clip(preds, 1.0, 3.0)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        mae = mean_absolute_error(targets, preds)
        return rmse, mae

    # Evaluate on Train/Val splits
    tr_rmse, tr_mae = get_metrics(DataLoader(CharDataset(X_s_tr, X_d_tr, y_tr), batch_size=BATCH_SIZE))
    val_rmse, val_mae = get_metrics(DataLoader(CharDataset(X_s_val, X_d_val, y_val), batch_size=BATCH_SIZE))
    
    # 6. Test Prediction
    print("Generating Test predictions...")
    test_loader = DataLoader(CharDataset(X_s_te, X_d_te, labels=None), batch_size=BATCH_SIZE)
    
    test_preds = []
    with torch.no_grad():
        for s, d in test_loader:
            s, d = s.to(device), d.to(device)
            out = model(s, d)
            test_preds.extend(out.cpu().numpy())
            
    test_preds = np.clip(test_preds, 1.0, 3.0)
    np.save(SUBMISSION_FILE, test_preds)
    print(f"Test predictions saved to '{SUBMISSION_FILE}'")
    
    # Log Results
    log_results(total_time, tr_rmse, val_rmse, tr_mae, val_mae)

if __name__ == "__main__":
    main()
