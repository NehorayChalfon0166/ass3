import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import time
import os

# --- Constants ---
RESULTS_FILE = 'results_log.csv'
MODEL_FILE = 'outputs/benchmark_model.pkl'
DATA_FILE = 'data/benchmark_data.pkl'
SUBMISSION_FILE = 'outputs/submission_benchmark.npy'

def evaluate(y_true, y_pred, name="Set"):
    y_pred = np.clip(y_pred, 1.0, 3.0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"  {name} RMSE: {rmse:.4f}")
    print(f"  {name} MAE:  {mae:.4f}")
    return rmse, mae

def log_results(model_name, runtime, train_rmse, val_rmse, train_mae, val_mae):
    # Logs TRAIN/VAL results only. Test results handled in evaluate.py
    file_exists = os.path.exists(RESULTS_FILE)
    
    # We leave Test columns empty or N/A for this script
    df = pd.DataFrame([{
        'Model type': model_name,
        'runtime': f"{runtime:.2f} sec",
        'Train RMSE': train_rmse,
        'Val-RMSE': val_rmse,
        'Test-RMSE': "N/A (See evaluate.py)",
        'Train MAE': train_mae,
        'Val-MAE': val_mae,
        'Test-MAE': "N/A (See evaluate.py)"
    }])
    
    if not file_exists:
        df.to_csv(RESULTS_FILE, index=False)
    else:
        df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    
    print(f"\nTraining Results saved to {RESULTS_FILE}")

def run_benchmark():
    print("--- Training Benchmark Model (Ridge Regression) ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run step1_preprocess.py first.")
        return

    print("Loading data...")
    data = joblib.load(DATA_FILE)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    
    # 2. Train Model
    print(f"Training Ridge Regression ({X_train.shape[0]} samples)...")
    start_time = time.time()
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Save Model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

    # 3. Predict & Evaluate (Train/Val only)
    print("\nEvaluating on Train/Val...")
    
    # Train
    y_pred_train = model.predict(X_train)
    train_rmse, train_mae = evaluate(y_train, y_pred_train, "Train")
    
    # Val
    y_pred_val = model.predict(X_val)
    val_rmse, val_mae = evaluate(y_val, y_pred_val, "Val")
    
    # Test (Prediction Only)
    print("Generating Test predictions...")
    y_pred_test = model.predict(X_test)
    y_pred_test = np.clip(y_pred_test, 1.0, 3.0)
    
    # Save predictions
    np.save(SUBMISSION_FILE, y_pred_test)
    print(f"Test predictions saved to '{SUBMISSION_FILE}'")
    
    # 4. Log Results
    log_results(
        "Naive Benchmark (Ridge Char 2-4gram)", 
        train_time,
        round(train_rmse, 4), 
        round(val_rmse, 4), 
        round(train_mae, 4), 
        round(val_mae, 4)
    )

if __name__ == "__main__":
    run_benchmark()