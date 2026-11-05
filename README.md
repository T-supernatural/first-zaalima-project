# Zalima Financial Transaction Fraud Detection

A focused reconstruction of a Data Science and Machine Learning internship project. This is a learning portfolio project, not a claim to reproduce an exact missing notebook or historical internship results. The aim is to identify potentially fraudulent transactions and explain the precision–recall tradeoff.

## Current status and actual results

The repository originally contained no project files or dataset. The notebook, reusable prediction code and optional Streamlit interface have been implemented. **No real-data training has been performed, no trained artifact is included, and no model scores are claimed.** The dataset must be obtained before training. Python was not available on PATH in the initial environment; runtime verification is reported separately when available.

Run the notebook to generate actual CV comparisons, held-out metrics, a confusion matrix, ROC and precision–recall curves, a saved model, and run metadata. Its final selection statement names the selected model using measured CV average precision. No model is declared the winner in advance.

## Setup (Python 3.11 or 3.12)

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Select the `.venv` Python kernel, open `notebooks/fraud_detection.ipynb`, and choose **Run All**. On macOS/Linux, use `.venv/bin/python` instead. If `python` is unavailable on Windows, install Python 3.12 with PATH enabled, or use `py -3.12` to create the environment. Dependency ranges allow compatible updates; each actual training run records core package versions in its metadata rather than claiming a fully locked environment.

For unattended execution after setup and download:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/fraud_detection.ipynb
```

## Dataset

Download the original [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) through the browser and extract `creditcard.csv` into `data/`. Kaggle may require sign-in and acceptance of its terms. Do not substitute similarly named balanced or resampled datasets. The notebook validates the schema and derives all statistics from the supplied file.

No API credentials are needed in the project for browser downloading. If you choose Kaggle's CLI, configure credentials securely following Kaggle's own instructions outside this repository. Never paste tokens in chat or commit them. Dataset files, archives, model binaries, local environments and generated reports are ignored by Git.

Expected columns:

| Columns | Meaning |
| --- | --- |
| `Time` | Elapsed time in seconds relative to the dataset's first transaction |
| `V1`–`V28` | Anonymised PCA features supplied by the dataset |
| `Amount` | Transaction amount |
| `Class` | Target: 0 legitimate, 1 fraudulent |

The project does not accept raw card numbers or generate the original PCA transformation. Counts, missing values, types and duplicates are shown when you run the notebook.

## Methodology

1. Inspect and validate the real data; remove and report exact duplicates before splitting. Removing duplicates prevents identical records crossing partitions but can discard genuine repeats.
2. Reserve a stratified 20% test set, seed 42. EDA of transaction patterns uses only the 80% training partition.
3. Put median imputation inside every pipeline, with standard scaling for logistic regression. Cross-validation fits preprocessing separately in each training fold.
4. Compare balanced logistic regression, a depth-limited decision tree and histogram-based XGBoost with three-fold stratified CV. XGBoost computes balanced sample weights inside each fit from that fold's labels. There is no oversampling or weighting based on test labels.
5. Rank baseline families by CV average precision. Tune the leading family with a four-configuration GridSearchCV, scored by average precision, run sequentially. This is 12 fold fits plus refit; the baseline comparison adds 9 fold fits. Runtime depends on laptop resources and convergence.
6. Fix the selected configuration from training CV, retain the baseline if tuning does not improve it, then evaluate the fixed choice and baseline models on the test partition. Do not change the selection after reading test results.
7. Save the selected training-fitted pipeline with Joblib. Save metrics, dataset SHA-256, versions, split sizes, seed and selected parameters under `reports/`.

Precision, recall, F1 and confusion matrices use threshold 0.5. ROC-AUC and **average precision (AP)** use scores. AP is a precision–recall summary, not trapezoidal PR area; accuracy is not the primary metric. Balanced training changes score calibration: the output is a fraud score, not a verified probability of fraud. A higher AP does not guarantee a better recall at threshold 0.5.

## Prediction and demonstration

After training, create a CSV containing exactly the 30 feature columns (`Time,V1,...,V28,Amount`) and at least one transaction, **without `Class`**. Do not put the literal ellipsis in a CSV header. The app supplies a complete downloadable header. Column order is normalised; missing feature values use training medians. Invalid strings, infinite values, extra/missing columns and negative Time/Amount are rejected.

```powershell
.\.venv\Scripts\python.exe -m src.predict path\to\transactions.csv
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The notebook demonstrates prediction using a genuine held-out row after removing its label, and checks a saved/reloaded pipeline against the in-memory scores. Python callers can use `from src.predict import predict_transactions` with a pandas DataFrame. The output contains `fraud_score` and `prediction`; prediction 1 flags potential fraud, 0 falls below 0.5. Joblib can execute code on load: use only trusted artifacts you trained or obtained from a trusted source.

## Files and generated outputs

- `notebooks/fraud_detection.ipynb`: the main deliverable, with all 16 requested sections.
- `src/predict.py`: input validation, model loading, inference and CSV CLI.
- `src/training.py`: importable XGBoost estimator with fold-local balanced weights.
- `app.py`: optional CSV upload demonstration, explicitly educational.
- `models/fraud_pipeline.joblib`: generated after training, includes preprocessing.
- `reports/cv_comparison.csv`: measured baseline CV comparison.
- `reports/test_metrics.json`: measured held-out comparison, including confusion matrices.
- `reports/run_metadata.json`: selected model, final metrics and reproducibility context.

Notebook outputs may contain real transaction feature values. Review outputs before sharing or committing. Nothing has been committed or pushed as part of this reconstruction.

## Explaining this project in an interview

Explain why rare fraud makes accuracy misleading, how precision measures investigation efficiency and recall measures captured fraud, and why average precision ranks models across thresholds. Point to the pipeline to explain leakage prevention. Explain that cross-validation chooses the model, while the held-out test estimates performance of the fixed choice. Use the actual generated confusion matrix to discuss missed fraud and false alerts. Do not present unexecuted code as measured results.

## Limitations and next steps

This is an educational demonstration, not a production financial security system. A random split cannot establish future performance or customer independence. The supplied PCA features were created outside this project. Deduplication changes the evaluation population. Model selection on shared CV folds can overstate CV performance; the untouched test set is the independent check. AP depends on fraud prevalence and rare-positive metrics can have substantial uncertainty.

Future work includes time-based evaluation, confidence intervals, calibration and cost-based threshold selection on separate validation data, drift monitoring and human review. SQL is omitted because the source is a single CSV and a database would add little here. No synthetic dataset or invented scores are supplied.

References: [scikit-learn AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).
