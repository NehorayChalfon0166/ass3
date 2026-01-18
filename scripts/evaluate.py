import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Configuration ---
SOLUTION_FILE = 'data/solution.csv'
RESULTS_FILE = 'results_log.csv'
PRED_FILES = {
    'Benchmark': 'outputs/submission_benchmark.npy',
    'Siamese Char CNN': 'outputs/submission_siamese_char.npy',
    'FE (Char) + XGBoost': 'outputs/submission_char_fe_xgb.npy',
    'FE (Char) + Ridge': 'outputs/submission_char_fe_ridge.npy'
}

def load_solution():
    if not os.path.exists(SOLUTION_FILE):
        print(f"Error: {SOLUTION_FILE} not found.")
        return None, None
    
    df = pd.read_csv(SOLUTION_FILE)
    
    # Filter Usage
    if 'Usage' in df.columns:
        # Keep Public and Private, ignore Ignored
        # The user said "remove -1", which usually corresponds to Ignored.
        mask = df['Usage'] != 'Ignored'
        # Double check with relevance just in case
        if 'relevance' in df.columns:
             mask = mask & (df['relevance'] != -1)
        
        filtered_df = df[mask]
        valid_indices = df.index[mask].to_numpy() # Original indices to slice predictions
        y_true = filtered_df['relevance'].values
        return y_true, valid_indices
    elif 'relevance' in df.columns:
        # Fallback if Usage column missing but -1 exists
        mask = df['relevance'] != -1
        filtered_df = df[mask]
        valid_indices = df.index[mask].to_numpy()
        y_true = filtered_df['relevance'].values
        return y_true, valid_indices
        
    return None, None

def evaluate_predictions(name, pred_file, y_true, valid_indices):
    if not os.path.exists(pred_file):
        print(f"[{name}] Prediction file {pred_file} not found.")
        return None, None

    try:
        y_pred_full = np.load(pred_file)
        
        # Check length
        # We assume y_pred_full corresponds to the full test/solution file
        # If lengths match full solution, we slice.
        # Note: valid_indices corresponds to the row number in solution.csv (0-based)
        
        if len(y_pred_full) < np.max(valid_indices):
             print(f"[{name}] Warning: Prediction length {len(y_pred_full)} < Max Index {np.max(valid_indices)}.")
             return None, None
             
        y_pred = y_pred_full[valid_indices]
        
        # Clip just in case
        y_pred = np.clip(y_pred, 1.0, 3.0)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"--- {name} ---")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        
        return rmse, mae
    except Exception as e:
        print(f"Error evaluating {name}: {e}")
        return None, None

def update_log(name, rmse, mae):
    if not os.path.exists(RESULTS_FILE):
        print("Results log not found, cannot update.")
        return

    df = pd.read_csv(RESULTS_FILE)
    
    # Mapping display names to Model type names in the CSV
    name_map = {
        'Benchmark': 'Naive Benchmark',
        'Siamese Char CNN': 'Character level CNN',
        'FE (Char) + XGBoost': 'FE (Char) + XGBoost',
        'FE (Char) + Ridge': 'FE (Char) + Ridge'
    }
    
    csv_name = name_map.get(name, name)
    
    updated = False
    for i in range(len(df)):
        if csv_name in str(df.loc[i, 'Model type']):
            # If we find a row for this model, we update it.
            # We prefer updating rows that have "N/A" for Test metrics.
            if df.loc[i, 'Test-RMSE'] == "N/A (See evaluate.py)" or pd.isna(df.loc[i, 'Test-RMSE']):
                df.loc[i, 'Test-RMSE'] = round(rmse, 4)
                df.loc[i, 'Test-MAE'] = round(mae, 4)
                updated = True
                print(f"Updated existing entry for {df.loc[i, 'Model type']}")
                break

    if not updated:
        # Check if any row matches exactly, even if it's already filled
        for i in range(len(df)):
            if csv_name == str(df.loc[i, 'Model type']):
                df.loc[i, 'Test-RMSE'] = round(rmse, 4)
                df.loc[i, 'Test-MAE'] = round(mae, 4)
                updated = True
                print(f"Overwrote existing entry for {df.loc[i, 'Model type']}")
                break
    
    if not updated:
        print(f"Could not find entry for {name} in {RESULTS_FILE}. Appending new row.")
        new_row = {
            'Model type': csv_name,
            'runtime': "-",
            'Train RMSE': "-",
            'Val-RMSE': "-",
            'Test-RMSE': round(rmse, 4),
            'Train MAE': "-",
            'Val-MAE': "-",
            'Test-MAE': round(mae, 4)
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(RESULTS_FILE, index=False)

def main():
    print("Loading Ground Truth from solution.csv...")
    y_true, valid_indices = load_solution()
    
    if y_true is None:
        print("Could not load valid labels.")
        return
    
    print(f"Found {len(y_true)} valid test samples (filtered 'Ignored/-1').")
    
    for name, pred_file in PRED_FILES.items():
        rmse, mae = evaluate_predictions(name, pred_file, y_true, valid_indices)
        if rmse is not None:
            update_log(name, rmse, mae)

    print(f"\nEvaluation Complete. Updated {RESULTS_FILE}.")

if __name__ == "__main__":
    main()