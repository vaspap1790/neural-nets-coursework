# Telco Customer Churn Prediction

Coursework for **Neural Nets and Deep Learning (AG704)**, MSc Artificial Intelligence (Aegean College / University of Essex).

Acting as a data scientist at a telecom company, this project builds, tunes and evaluates a TensorFlow/Keras
neural network to predict customer churn, benchmarked against a Logistic Regression baseline, Random Forest
and XGBoost, with SHAP-based explainability. The accompanying report is `Vasileios_Papadimitrakopoulos_NN_and_DL.docx`
(outside this repository).

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM, via Kaggle/GitHub),
7,043 customers, 21 columns. Expected at `data/Telco-Customer-Churn.csv` (not committed — see `.gitignore`).

## Project structure

```
src/                     Shared, reusable pipeline code
  data_utils.py            Dataset loading and column-schema constants
  preprocessing.py         Cleaning, encoding, train/test split, scaling
  baseline_model.py         Logistic Regression baseline
  nn_model.py                Keras MLP with residual blocks, BatchNorm, Dropout, L1/L2
  tuning.py                   KerasTuner (RandomSearch) hyperparameter search space
  explainability.py          SHAP values and summary plot
  benchmarking.py             Random Forest and XGBoost
  evaluation.py                Metrics, confusion matrix, ROC curve, metrics table

scripts/                 One entry point per coursework question
  q1_eda_preprocessing.py     Dataset exploration and preprocessing
  q2_baseline_model.py        Logistic Regression baseline
  q3_neural_network.py        Neural network: tuning, training, SHAP
  q4_analysis_benchmarking.py Random Forest / XGBoost benchmarking

outputs/
  figures/                 Generated plots (fig1-fig11)
  tables/                  Generated CSV tables

data/                     Raw dataset (gitignored)
tests/                    Reserved for unit tests
```

Each `scripts/qN_*.py` imports only from `src/`, writes its figures to `outputs/figures/` and its tables to
`outputs/tables/`, and can be run independently as long as the earlier questions' scripts (and therefore their
output files) have been run at least once — Q3 and Q4 re-derive the data split themselves, so strict ordering
is not required, but running `q1 → q2 → q3 → q4` matches how the report was produced.

## Setup

TensorFlow/KerasTuner/XGBoost need Python 3.11-3.12 (this project was built against Python 3.12; TensorFlow
does not yet support 3.13 on Apple Silicon at the time of writing). On macOS, XGBoost also needs `libomp`:

```bash
brew install libomp          # macOS only, required by xgboost
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `Telco-Customer-Churn.csv` in `data/` before running anything.

## Running

```bash
source .venv/bin/activate
python scripts/q1_eda_preprocessing.py
python scripts/q2_baseline_model.py
python scripts/q3_neural_network.py      # runs KerasTuner search; a few minutes
python scripts/q4_analysis_benchmarking.py
```

## Results summary

Test-set metrics (20% stratified hold-out, 1,409 customers):

| Metric    | Logistic Regression | Neural Network | Random Forest | XGBoost |
|-----------|---------------------|----------------|----------------|---------|
| Accuracy  | 0.7381 | 0.7189 | 0.7715 | 0.7516 |
| Precision | 0.5043 | 0.4833 | 0.5594 | 0.5213 |
| Recall    | 0.7834 | 0.8503 | 0.6551 | 0.7861 |
| F1-score  | 0.6136 | 0.6163 | 0.6034 | 0.6269 |
| ROC-AUC   | 0.8417 | 0.8355 | 0.8250 | 0.8388 |

No single model wins on every metric; the full comparison, error-cost analysis and business recommendations
are in the report. The neural network's own metrics are reported from one fixed, logged run rather than
re-derived on every execution — see the module docstring in `scripts/q4_analysis_benchmarking.py` for why
(TensorFlow training has run-to-run variance even with a fixed seed, due to CPU-level non-determinism).

## Notes

- `src/preprocessing.py` fits `StandardScaler` on the training fold only, to avoid data leakage.
- Class imbalance (~26.5% churn) is handled via `class_weight='balanced'` (Logistic Regression, neural
  network) and `scale_pos_weight` (XGBoost).
- SHAP values are computed with a permutation-based `Explainer` (not `DeepExplainer`), since the model is
  called through a generic Keras `predict()` wrapper.
