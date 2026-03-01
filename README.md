# Job Satisfaction Prediction with Ordinal Logistic Regression

This repository showcases an end-to-end machine learning workflow for predicting ordinal job satisfaction levels from large-scale developer survey data. The main artifact is the notebook:

- `job_satisfaction_ordinal_modeling.ipynb`

The notebook is structured to be readable on GitHub and demonstrates the full lifecycle from raw tabular data preparation to model evaluation and interpretation.

## Project Goal

Build an ordinal classification pipeline to predict `JobSat` (0-10 scale) using demographic, career, workplace, and AI-adoption related features.

## Dataset

Due to repository size restrictions, the complete original survey files are not stored in this project.

The full dataset (Stack Overflow Developer Survey 2025) can be downloaded from the official source:
- https://survey.stackoverflow.co/2025/

For this notebook, the raw survey data was preprocessed in advance and saved as:
- `processed_data.csv`

## What the Notebook Covers

1. Data loading and initial exploration
2. Missing-value analysis and imputation strategy
3. Categorical encoding:
   - Binary mapping (Yes/No -> 1/0)
   - One-hot encoding for nominal variables
   - Ordered encoding for ordinal variables
4. Multi-response feature handling (semicolon-separated answers)
5. Feature engineering:
   - Composite workplace and experience indicators
   - Aggregated behavior and technology-choice features
6. Feature selection using Random Forest importance
7. Custom ordinal logistic regression implementation (k-1 threshold models)
8. Model validation:
   - Stratified 10-fold cross-validation
   - Bias-variance diagnostics
   - Hyperparameter tuning via GridSearchCV
9. Final train/test evaluation and error pattern analysis

## Key Outcomes

- Dataset size after loading: **26,643 rows**, **49 initial columns**
- Feature space reduced to **79 selected features** after engineering/selection
- Best tuned model (ordinal logistic regression):
  - `C=1`, `penalty='l2'`, `class_weight=None`, `max_iter=100`
  - Best CV `f1_macro`: **0.0977**
- Final held-out test performance:
  - Accuracy: **0.2559**
  - Macro F1: **0.1035**

## Why This Project Matters

This notebook demonstrates practical handling of real-world survey data issues:

- high-cardinality categorical features
- mixed ordinal/nominal/multi-select responses
- class imbalance in ordinal targets
- balancing interpretability and predictive performance

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- seaborn, matplotlib

## Requirements

Install dependencies listed in:
- `requirements.txt`

Example:
```bash
pip install -r requirements.txt
```

## How to Run

1. Place the notebook in a folder with the required CSV data files.
2. Install dependencies from `requirements.txt`.
3. Open `job_satisfaction_ordinal_modeling.ipynb` in Jupyter Notebook/Lab.
4. Run cells top-to-bottom.

Notes:
- The notebook is intentionally output-cleaned for GitHub readability.
- Plots and metrics will regenerate on execution.
