# GoEmotions Sentiment Classification Using Classical ML Models

This project benchmarks multiple classical machine learning models on the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset after mapping emotion labels to sentiment classes.

## What This Project Does

- Loads and merges `goemotions_1.csv`, `goemotions_2.csv`, and `goemotions_3.csv`.
- Cleans and normalizes text.
- Maps multilabel emotions to sentiment categories.
- Builds vector features using Bag-of-Words and TF-IDF.
- Applies feature selection.
- Trains and compares multiple classifiers.
- Evaluates with macro metrics (`accuracy`, `precision`, `recall`, `f1_score`).
- Supports imbalanced-data handling variants:
  - SMOTE oversampling
  - Random undersampling
  - Stratified K-Fold voting setup

## Repository Structure

```text
.
|-- main.py
|-- Keeping_neutral_mixed.py
|-- confusion_matrix.py
|-- comparison_results_*.csv
|-- data/
|   |-- raw/
|   |   |-- goemotions_1.csv
|   |   |-- goemotions_2.csv
|   |   `-- goemotions_3.csv
|   |-- sentiment_dict.json
|   |-- ekman_mapping.json
|   `-- emotions.txt
|-- notebook/
|   |-- 01_eda.ipynb
|   |-- 02_preprocessing.ipynb
|   `-- 03_baseline.ipynb
`-- src/
    |-- preprocessing/
    |-- vectorization/
    |-- models/
    |-- evaluation/
    `-- training/
```

## Labeling Variants

### Default sentiment mapping (`target_preprocessing.py`)

Records are mapped to `positive`, `negative`, or `ambiguous`.

- Rows mapped to `neutral` or `mixed` are removed.
- Numeric labels:
  - `ambiguous -> 0`
  - `positive -> 1`
  - `negative -> 2`

### Neutral/Mixed kept (`target_preprocessing_neutral_mixed.py`)

Rows with `neutral` and `mixed` are retained.

- Numeric labels:
  - `neutral -> 0`
  - `positive -> 1`
  - `negative -> 2`
  - `ambiguous -> 3`
  - `mixed -> 4`

## Models Included

- Logistic Regression
- Random Forest
- XGBoost
- SGD Classifier
- KNN
- SVM (used in selected scripts)

## Vectorizers Included

- TF-IDF (`ngram_range=(1, 3)`, `max_features=5000`)
- Bag-of-Words (`ngram_range=(1, 3)`, `max_features=5000`)

## Setup

### 1) Create and activate a virtual environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Windows (cmd)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2) Install dependencies

```bash
pip install -U pip
pip install pandas numpy scikit-learn scipy matplotlib nltk contractions xgboost imbalanced-learn
```

### 3) Download NLTK resources

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## Running Experiments

### Main benchmark (SMOTE variant)

```bash
python main.py
```

Output:

- `comparison_results_4.csv`

### Keep neutral/mixed labels variant

```bash
python Keeping_neutral_mixed.py
```

Output:

- `comparison_results_neutral_mixed.csv`

### Confusion matrix example

```bash
python confusion_matrix.py
```

This trains a selected model and opens a confusion matrix plot window.

## Evaluation Modules

Reusable evaluation logic is under `src/evaluation/`:

- `compare.py`: baseline train/test split comparison.
- `compare_with_smote.py`: train/test split with SMOTE.
- `compare_with_RandomUnderSampler.py`: train/test split with random undersampling.
- `compare_with_K_Fold.py`: stratified K-Fold setup with prediction voting.
- `metrics.py`: common macro metrics.

## Notes and Caveats

- `xgboost` may require additional build tools in some local environments; using an up-to-date `pip` helps.
- The project assumes dataset files already exist in `data/raw/`.
- Current scripts are designed as executable scripts rather than CLI modules.

## Typical Workflow

1. Validate dataset files in `data/raw/`.
2. Run `main.py` for the primary benchmark.
3. Compare generated CSV results across variants.
4. Use notebooks in `notebook/` for EDA and baseline exploration.

## Future Improvements (Optional)

- Add a `requirements.txt` for reproducible installs.
- Add a single CLI entrypoint with flags (vectorizer/model/sampling).
- Add model persistence and experiment tracking.
- Add automated tests for preprocessing and label mapping.

## License

Add your preferred license in a `LICENSE` file.
