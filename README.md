# Home Depot Search Relevance - Character Level Models (Part 1)

This repository contains the Character-Level Siamese Network and Benchmarking code for Assignment 3.

## Project Structure

*   **`data/`**: Place your input dataset files here.
    *   *Required:* `train.csv`, `test.csv`, `product_descriptions.csv`
    *   *Generated:* `dl_data.npz` (Character sequences), `benchmark_data.pkl` (N-grams)
*   **`scripts/`**: Source code for training and evaluation.
    *   `step1_preprocess.py`: Tokenizes text into characters.
    *   `train_benchmark.py`: Trains Ridge Regression on N-grams.
    *   `train_model.py`: Trains the Siamese Char-CNN.
    *   `step4_char_fe.py`: Extracts features from the trained CNN and trains XGBoost/Ridge.
    *   `evaluate.py`: Calculates final metrics using `solution.csv`.
*   **`outputs/`**: Contains trained models, prediction files, and plots.
    *   `siamese_char_cnn.pt`: Trained PyTorch model.
    *   `training_history_char_siamese.png`: Loss curve.
*   **`run_pipeline.py`**: **Run this script to execute the full workflow.**

## Setup Instructions

1.  **Install Dependencies:**
    ```bash
    pip install torch pandas numpy scikit-learn xgboost matplotlib joblib
    ```

2.  **Prepare Data:**
    Download the Home Depot dataset and place the CSV files inside the `data/` folder.

3.  **Run Pipeline:**
    Execute the master script. It will automatically detect which steps are done and resume from where it left off.
    ```bash
    python run_pipeline.py
    ```

## Results

Current results on the Test set:

| Model | Test RMSE |
| :--- | :--- |
| **Naive Benchmark** | 0.6379 |
| **Siamese Char CNN** | **0.5310** |
| Feature Extraction (XGB) | 0.5503 |

See `REPORT_DRAFT.md` for the detailed report text.
