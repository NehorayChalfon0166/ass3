import subprocess
import sys
import time
import os

def run_command(command, description, output_files=None):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    
    # Check if outputs exist
    if output_files:
        if isinstance(output_files, str):
            output_files = [output_files]
        
        all_exist = True
        for f in output_files:
            if not os.path.exists(f):
                all_exist = False
                break
        
        if all_exist:
            print(f"SKIPPING: Output files ({', '.join(output_files)}) already exist.")
            return

    print(f"CMD : {command}")
    print(f"{ '='*60}")
    
    start = time.time()
    try:
        # Run using the current python interpreter
        full_cmd = f"{sys.executable} {command}"
        result = subprocess.run(full_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n!!! ERROR in step '{description}' !!!")
        print(f"Exit Code: {e.returncode}")
        sys.exit(1)
    
    elapsed = time.time() - start
    print(f"\n>>> Step '{description}' completed in {elapsed:.2f} seconds.")

def main():
    print("Starting End-to-End Pipeline for Assignment 3 (Part 1)")
    
    # 1. Preprocess
    run_command("scripts/step1_preprocess.py", "Data Preprocessing & Tokenization", 
                output_files=["data/dl_data.npz", "data/benchmark_data.pkl"])
    
    # 2. Train Benchmark
    run_command("scripts/train_benchmark.py", "Train Naive Benchmark (Ridge)", 
                output_files=["outputs/benchmark_model.pkl", "outputs/submission_benchmark.npy"])
    
    # 3. Train Siamese Model
    run_command("scripts/train_model.py", "Train Character-Level Siamese Model", 
                output_files=["outputs/siamese_char_cnn.pt", "outputs/submission_siamese_char.npy"])
    
    # 4. Feature Extraction (Char)
    run_command("scripts/step4_char_fe.py", "Feature Extraction (Char Model + XGB/Ridge)", 
                output_files=["outputs/submission_char_fe_xgb.npy", "outputs/submission_char_fe_ridge.npy"])
    
    # 5. Evaluate (Always run to ensure log is up to date)
    run_command("scripts/evaluate.py", "Final Evaluation & Report Generation")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{ '='*60}")
    print("Artifacts generated:")
    print("1. data/dl_data.npz, data/benchmark_data.pkl (Data)")
    print("2. outputs/benchmark_model.pkl, outputs/siamese_char_cnn.pt (Models)")
    print("3. outputs/submission_*.npy (Predictions for Benchmark, Siamese, XGB, Ridge)")
    print("4. results_log.csv (Metrics Table)")
    print("5. outputs/training_history_char_siamese.png (Training Plot)")
    print("6. outputs/tokenization_examples.txt (Tokenization Samples)")

if __name__ == "__main__":
    main()
