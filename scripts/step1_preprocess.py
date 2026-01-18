import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

# --- Constants based on your statistics ---
MAX_LEN_SEARCH = 64    # Covers Max (60)
MAX_LEN_DESC = 2048    # Covers >95% (1842)
RANDOM_SEED = 42

# --- Paths ---
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'

def load_and_merge():
    print("Loading CSV files...")
    # Load with fallback encoding
    try:
        train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), encoding='ISO-8859-1')
        test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), encoding='ISO-8859-1')
        desc = pd.read_csv(os.path.join(DATA_DIR, 'product_descriptions.csv'), encoding='ISO-8859-1')
    except:
        train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), encoding='latin-1')
        test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), encoding='latin-1')
        desc = pd.read_csv(os.path.join(DATA_DIR, 'product_descriptions.csv'), encoding='latin-1')

    # Merge descriptions
    print("Merging descriptions...")
    train = pd.merge(train, desc, on='product_uid', how='left')
    test = pd.merge(test, desc, on='product_uid', how='left')
    
    # Fill NaNs
    train['search_term'] = train['search_term'].fillna("")
    train['product_description'] = train['product_description'].fillna("")
    train['product_title'] = train['product_title'].fillna("")
    
    test['search_term'] = test['search_term'].fillna("")
    test['product_description'] = test['product_description'].fillna("")
    test['product_title'] = test['product_title'].fillna("")
    
    # Create combined description (Title + Description) for better context
    train['full_desc'] = train['product_title'] + " " + train['product_description']
    test['full_desc'] = test['product_title'] + " " + test['product_description']
    
    return train, test

def build_char_dict(texts):
    print("Building character dictionary...")
    chars = set()
    for text in texts:
        chars.update(text)
    
    # 0 is reserved for padding
    char_to_int = {c: i + 1 for i, c in enumerate(sorted(list(chars)))}
    return char_to_int

def text_to_sequence(text, char_to_int, max_len):
    seq = [char_to_int.get(c, 0) for c in text]
    # Truncate
    seq = seq[:max_len]
    # Pad (Post-padding usually fine, or pre-padding)
    # We will use zeros for padding
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    return seq

def prepare_dl_data(train_df, test_df):
    print("\n--- Preparing Data for Deep Learning (Character Sequences) ---")
    
    # 1. Build Vocabulary from all text
    all_text = pd.concat([
        train_df['search_term'], 
        train_df['full_desc'],
        test_df['search_term'],
        test_df['full_desc']
    ])
    char_to_int = build_char_dict(all_text)
    print(f"Vocabulary Size: {len(char_to_int)} characters")
    
    # 2. Convert to Sequences
    print("Converting text to sequences (this may take a moment)...")
    
    def process_column(series, max_len):
        return np.array([text_to_sequence(t, char_to_int, max_len) for t in series], dtype=np.int8)
    
    X_train_search = process_column(train_df['search_term'], MAX_LEN_SEARCH)
    X_train_desc = process_column(train_df['full_desc'], MAX_LEN_DESC)

    # --- Save Tokenization Examples for Report ---
    token_file = os.path.join(OUTPUT_DIR, 'tokenization_examples.txt')
    print(f"Saving tokenization examples to '{token_file}'...")
    with open(token_file, 'w', encoding='utf-8') as f:
        f.write("--- Tokenization Examples ---\n")
        f.write(f"Vocabulary Size: {len(char_to_int)}\n")
        f.write("Char to Int Map (First 20): " + str(list(char_to_int.items())[:20]) + "...\n\n")
        
        for i in range(5):
            orig = train_df['search_term'].iloc[i]
            seq = X_train_search[i]
            # Convert non-zero seq back to readable for check (optional) 
            f.write(f"Example {i+1}:\n")
            f.write(f"  Original: '{orig}'\n")
            f.write(f"  Sequence: {seq.tolist()[:20]} ... (truncated)\n\n")
    # ---------------------------------------------
    
    X_test_search = process_column(test_df['search_term'], MAX_LEN_SEARCH)
    X_test_desc = process_column(test_df['full_desc'], MAX_LEN_DESC)
    
    y = train_df['relevance'].values
    
    # 3. Split Train into Train/Validation
    print("Splitting Training data into Train/Validation (80/20)...")
    # We split indices to keep pairs together
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED)
    
    dl_data = {
        'X_train_search': X_train_search[train_idx],
        'X_train_desc': X_train_desc[train_idx],
        'y_train': y[train_idx],
        
        'X_val_search': X_train_search[val_idx],
        'X_val_desc': X_train_desc[val_idx],
        'y_val': y[val_idx],
        
        'X_test_search': X_test_search,
        'X_test_desc': X_test_desc,
        
        'char_to_int': char_to_int,
        'max_len_search': MAX_LEN_SEARCH,
        'max_len_desc': MAX_LEN_DESC
    }
    
    np.savez(os.path.join(DATA_DIR, 'dl_data.npz'), **dl_data)
    print("Deep Learning data saved to 'dl_data.npz'")

def prepare_benchmark_data(train_df, test_df):
    print("\n--- Preparing Data for Benchmark (CountVectorizer n-gram 2,4) ---")
    
    # Combine Search + Desc for Bag-of-Ngrams
    # We add a separator
    train_text = train_df['search_term'] + " \t " + train_df['full_desc']
    test_text = test_df['search_term'] + " \t " + test_df['full_desc']
    
    # Setup Vectorizer (As requested: ngram 2-4)
    # limit max_features to avoid OOM
    print("Fitting CountVectorizer (char, ngram 2-4)...")
    vectorizer = CountVectorizer(
        analyzer='char',
        ngram_range=(2, 4),
        min_df=5,       # Ignore very rare ngrams to save memory
        dtype=np.uint16,
        max_features=20000
    )
    
    # Fit on Train, Transform both
    # Note: Fitting on just train is standard practice to avoid data leakage
    X_train_full = vectorizer.fit_transform(train_text)
    X_test_full = vectorizer.transform(test_text)
    
    y = train_df['relevance'].values
    
    # Split
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_SEED)
    
    benchmark_data = {
        'X_train': X_train_full[train_idx],
        'y_train': y[train_idx],
        'X_val': X_train_full[val_idx],
        'y_val': y[val_idx],
        'X_test': X_test_full,
        'vectorizer': vectorizer
    }
    
    joblib.dump(benchmark_data, os.path.join(DATA_DIR, 'benchmark_data.pkl'))
    print("Benchmark data saved to 'benchmark_data.pkl'")

if __name__ == "__main__":
    train_df, test_df = load_and_merge()
    
    prepare_dl_data(train_df, test_df)
    prepare_benchmark_data(train_df, test_df)
    
    print("\nPreprocessing Complete!")