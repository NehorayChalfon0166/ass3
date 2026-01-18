# Assignment 3: Home Depot Search Relevance - Research Report

## 1. Preface

**Problem Description:**
The objective of this assignment was to predict the relevance score (on a scale of 1.0 to 3.0) of a product given a user's search term and the product's title/description. This is a core problem in Information Retrieval and E-commerce, directly impacting user experience and conversion rates. The challenge lies in capturing the semantic similarity between short, often noisy search queries and detailed product descriptions.

**Methods:**
We explored two primary approaches:
1.  **Character-Level Siamese Network:** A deep learning model that processes text as a sequence of characters. This approach handles out-of-vocabulary words robustly and captures morphological similarities (e.g., "bracket" vs. "brackets"). We used a 1D CNN as the encoder shared between the search term and description branches.
2.  **Word-Level Approaches (shahar Section):**

**Conclusions:**
Our experiments demonstrated that the **Character-level Siamese CNN (RMSE 0.5310)** significantly outperformed the Naive Benchmark (Ridge Regression on N-grams, RMSE 0.6379). This confirms that learning a dense, metric-based representation of text is superior to sparse, surface-level statistical methods. Interestingly, using the trained Siamese network merely as a feature extractor for XGBoost yielded slightly worse results (RMSE ~0.55) than the end-to-end model, suggesting that the joint optimization of the encoder and the regression head in the Siamese network is crucial for peak performance.

---

## 2. Methodology & Challenges (Research Process)

**Design Decisions:**
*   **N-gram Constraints:** Initially, we attempted to use larger N-gram ranges (up to 5 or 6) and unlimited vocabulary size for the Naive Benchmark to capture more context. However, this resulted in **Memory Errors (OOM)** due to the massive dimensionality of the character-based sparse matrix.
*   **Solution:** We restricted the model to `ngram_range=(2, 4)` and enforced a hard limit of `max_features=20,000` with `min_df=5`. This balanced memory usage with performance, providing a "good enough" baseline that didn't crash the system.

**Preprocessing:**
We implemented a robust preprocessing pipeline that merges product titles with descriptions to provide maximum context. We also ensured that the train/validation split preserved `product_uid` grouping to prevent "data leakage" (where the model sees the same product in both train and val, artificially inflating scores).

---

## 3. Tokenization Methods

### A. Character-Level Tokenization
For the character-based models, we treated the text as a sequence of individual characters. This allows the model to learn from the internal structure of words.

*   **Vocabulary:** We built a dictionary of all unique characters found in the corpus (125 unique characters).
*   **Encoding:** Each character was mapped to an integer ID.
*   **Padding:** Sequences were padded with zeros to a fixed length (64 for Search Terms, 2048 for Descriptions).

**Examples:**
*   Original: `'angle bracket'`
    *   Sequence: `[66, 79, 72, 77, 70, 1, 67, 83, 66, 68, 76, 70, 85, ...]`
*   Original: `'l bracket'`
    *   Sequence: `[77, 1, 67, 83, 66, 68, 76, 70, 85, ...]`
    *(Note: The common suffix ' bracket' [1, 67, 83...] is clearly visible in the numerical sequence)*

### B. Word-Level Tokenization
[shahar insert the Word2Vec/BERT tokenization examples here]

---

## 3. Training Process Metrics

### A. Character-Level Siamese CNN
*   **Model Architecture:** Shared Embedding Layer (64-dim) -> 3x Conv1d Layers with ReLU & MaxPool -> Fully Connected Layer.
*   **Loss Function:** Mean Squared Error (MSE).
*   **Optimizer:** Adam (LR=0.001).
*   **Training Plot:**
![Character Siamese CNN Training History](outputs/training_history_char_siamese.png)

*   **Observation:** The model converged quickly, with Validation RMSE stabilizing around epoch 5-7. The best model was saved at the point of lowest Validation RMSE (0.4803) to prevent overfitting.

*(Note: The Naive Benchmark and Feature Extraction models (Ridge/XGBoost) are non-iterative or tree-based solvers that do not produce "Loss vs. Epoch" training curves in the same manner as Neural Networks, hence no plots are included for them.)*

### B. Word-Level Model
[sahar insert the training plots here]

---

## 4. Results Comparison

The following table summarizes the performance of all tested methods. The **Test RMSE** is the primary metric for comparison.

| Model type | Runtime | Train RMSE | Val-RMSE | Test-RMSE | Test-MAE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Benchmark (Ridge)** | 873s | 0.4027 | 0.5944 | **0.6379** | 0.5065 |
| **Character Siamese CNN** | 1983s | 0.3854 | 0.4886 | **0.5310** | 0.4265 |
| **FE (Char) + XGBoost** | 61s | 0.3606 | 0.4926 | 0.5503 | 0.4394 |
| **FE (Char) + Ridge** | 6s | 0.4048 | 0.4923 | 0.5489 | 0.4387 |
| *Word Level Model* | ... | ... | ... | *[shahar]* | ... |
| *FE (Word) + XGBoost* | ... | ... | ... | *[shahar]* | ... |

*(FE = Feature Extraction)*

---

## 5. Final Remarks

**What has been learned:**
1.  **Characters matter:** In domains with technical terms, typos, or specific model numbers (common in hardware stores), character-level models provide a distinct advantage over word-level models that might struggle with "unknown" tokens.
2.  **End-to-End vs. Two-Stage:** Training the Siamese network end-to-end (optimizing the encoder for the specific regression task) proved more effective than treating it as a static feature extractor for a Gradient Boosting machine. The XGBoost model likely overfitted to the training features (Train RMSE 0.36 vs Test 0.55) compared to the more balanced Siamese model.
3.  **Data Filtering:** Properly handling the dataset (filtering out rows marked "Ignored" or "-1") was critical for accurate evaluation.

**Future Exploration:**
*   **Hybrid Models:** Combining Character-level embeddings (for morphology) with Word-level embeddings (BERT/Word2Vec for semantics) could offer the best of both worlds.
*   **Attention Mechanisms:** Adding an attention layer to weigh specific parts of the description heavily (e.g., dimensions or material type) based on the search query could further improve relevance scoring.
