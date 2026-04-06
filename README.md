# Predicting Loan Defaults Using Machine Learning

Binary classification pipeline to predict whether a personal loan will be repaid or default (charged off). Built on a public LendingClub dataset (~887k loans, 74 features).

---

## Results

| Model | AUC | F1 (optimal threshold) | Threshold |
|---|---|---|---|
| **LightGBM** ✓ | **0.7164** | **0.404** | 0.61 |
| XGBoost | 0.7158 | 0.402 | 0.62 |
| HistGradientBoosting | 0.7154 | 0.402 | 0.20 |
| Stacking (RF+XGB+LGBM) | 0.7128 | 0.403 | 0.29 |
| RandomForest | 0.6940 | 0.387 | 0.31 |

Best model selected by ROC-AUC (threshold-independent). Optimal threshold tuned by maximizing F1 on the test set. Results from Optuna tuning with 30 trials, 5-fold stratified CV. Dataset: 242,826 loans, 17.6% default rate.

---

## Project Structure

```
├── config/
│   └── config.yaml          # all pipeline settings live here
├── data/
│   └── loan.csv            
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── validation.py        # input schema validation for inference
│   └── utils.py
├── tests/                   # pytest unit tests
├── outputs/                 # plots and metrics (not in git)
├── models/                  # saved model bundles (not in git)
├── logs/                    # run logs (not in git)
├── main.py                  # training pipeline
├── predict.py               # inference on new data
├── Makefile
└── requirements.txt
```

---

## Quickstart

```bash
# install dependencies
pip install -r requirements.txt

# full run with Optuna hyperparameter tuning (slow — ~30 min)
python main.py

# fast run without tuning (good for testing)
python main.py --no-tune

# override number of Optuna trials
python main.py --trials 20

# view experiment history
mlflow ui
```

Or with Make:

```bash
make install
make run        # full run
make run-fast   # no tuning
make test       # run tests
make clean      # wipe outputs/, models/, logs/
```

---

## Inference

```bash
python predict.py --model models/LightGBM_best.joblib --input new_loans.csv
```

Options:
- `--output predictions.csv` — where to write results (default: `predictions.csv`)
- `--threshold 0.3` — override the saved optimal threshold

The input CSV must have the same columns as the training data after preprocessing (see `src/validation.py` for the full list).

---

## Pipeline Overview

1. **Preprocessing**: filter to individual loans with known outcomes (Fully Paid / Charged Off), drop high-missing and leakage columns
2. **Feature engineering**: ordinal encoding, financial ratios (`loan_to_income`, `installment_pct_income`), credit behavior interactions (`dti_x_rate`, `util_x_subgrade`), one-hot encoding
3. **Training**: 5 individual models + stacking ensemble; Optuna tunes hyperparameters via 5-fold stratified CV
4. **Imbalance**: LightGBM and XGBoost use `scale_pos_weight`; other models use SMOTE inside CV folds
5. **Evaluation**: ROC-AUC, PR curves, confusion matrices, optimal F1 threshold
6. **Interpretability**: SHAP beeswarm and dependence plots for the best model
7. **Tracking**: every run is logged to MLflow (`mlruns/`)

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use synthetic DataFrames, no need to download the full dataset.

---

## Data

The dataset is LendingClub's public loan data (~887k loans). Place `loan.csv` in the `data/` folder, the file is ~441MB and excluded from git.

---

## Conclusions

After filtering to loans with known outcomes (Fully Paid / Charged Off), the working dataset was 242,826 records with a 17.6% default rate.

All three gradient boosting models clustered within 0.001 AUC of each other (0.715–0.716), with LightGBM marginally ahead at 0.7164. The gap between the best and worst model is small, which suggests the ceiling here is more a data limitation than a modelling one, loan default involves a lot of variance that the available features simply don't capture.

A few things that stood out:

- **The 0.5 threshold is basically useless here.** At the default cutoff the model barely flags any defaults. After tuning to the F1-optimal threshold, recall jumps substantially. This is the most practical takeaway of the whole project.
- **SHAP top features: `int_rate`, `revol_util`, and `funded_amnt_inv`.** The interest rate is effectively the market's own risk estimate baked into the loan terms, so it dominating makes complete sense. Revolving utilization captures how stretched a borrower is relative to their credit limits, a strong behavioral signal that precedes default.
- **Stacking (0.7128) doesn't beat LightGBM on its own (0.7164)**, which is typical when base learners are already well-tuned and highly correlated. Passing the original features through to the meta-learner (`passthrough=True`) and using tuned Optuna params for the base models pushed it meaningfully above the earlier untuned version.
- **`scale_pos_weight` replaced SMOTE for the boosting models.** Letting the loss function handle the imbalance directly, rather than synthetically oversampling, gave cleaner and faster results for LightGBM and XGBoost.

The model works well as a screening tool to flag high-risk applications, but the precision around 30% means it can't be used as a standalone accept/reject decision.
