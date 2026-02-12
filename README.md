<p align="center">
  <img src="images/banner.png" alt="Project Banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/github/last-commit/choemily/office-leasing-churn-portfolio">
  <img src="https://img.shields.io/github/repo-size/choemily/office-leasing-churn-portfolio">
  <img src="https://img.shields.io/badge/Model-LogReg%20%7C%20LightGBM-orange.svg">
  <img src="https://img.shields.io/badge/Status-Active-success.svg">
</p>

# Office Leasing Churn Prediction (Portfolio Project)

> **Purpose**: A presentation of a **tenant churn prediction** workflow for office landlords. It includes data generation (synthetic), feature engineering, model training (**Logistic Regression** vs **Gradient Boosting**), evaluation, and reporting.

> **Note on Data**: For demonstration purpose, this project ships with a **synthetic tenant‑month dataset** (generated via `src/generate_synthetic_data.py`) to make the repo fully runnable without access to confidential lease data. 

---

## 🧭 What This Project Demonstrates
- How to **frame churn** in office leasing (non-renewal at expiry within 12 months)
- How to build a **tenant‑month panel** and key features (months to expiry, rent gap, arrears, tickets, etc.)
- Baseline **Logistic Regression** vs **LightGBM** comparison
- Business‑centric metrics: **Precision@TopK**, **ROC-AUC**, **Revenue at Risk**, and a ranked **risk list**
- Clear code structure, configuration via YAML, and reproducible runs

---

## 📁 Repository Structure
```
office-leasing-churn-portfolio/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ Makefile
├─ config/
│  └─ config.yaml                 # Hyperparameters & paths
├─ data/
│  ├─ raw/                        # (empty) place raw data here if available
│  └─ processed/                  # synthetic tenant_month.csv saved here
├─ notebooks/
│  └─ 01_eda_and_feature_profiling.py  # Jupytext-style notebook (optional)
├─ runs/                          # trained models & reports
└─ src/
   ├─ __init__.py
   ├─ generate_synthetic_data.py
   ├─ features.py
   ├─ models.py
   ├─ train.py
   ├─ evaluate.py
   └─ utils.py
```

---

## 🚀 Quickstart
```bash
# 0) Create & activate a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts ctivate

# 1) Install dependencies
pip install -r requirements.txt

# 2) Generate a synthetic tenant-month dataset (~50k rows by default)
python src/generate_synthetic_data.py --n_tenants 1500 --months 24 --out data/processed/tenant_month.csv

# 3) Train baseline Logistic Regression
python src/train.py --model logreg --config config/config.yaml --data data/processed/tenant_month.csv

# 4) Train Gradient Boosting (LightGBM)
python src/train.py --model lgbm --config config/config.yaml --data data/processed/tenant_month.csv

# 5) Evaluate and produce risk ranking (top-K)
python src/evaluate.py --runs_dir runs --k 50 --revenue_col revenue_at_risk
```

Artifacts (models, metrics, plots) will be stored under `runs/` with timestamps.

---

## 🧪 Labels & Target Definition
- **Churn label**: *non‑renewal at lease expiry within the next 12 months* from each reference month.
- The synthetic generator simulates a hazard that increases as **months‑to‑expiry** approaches and with higher **rent gap**, **late payments**, **service tickets**, and lower **satisfaction**.

---

## 📊 Metrics
- **ROC‑AUC**
- **Precision@K** (e.g., top 50 tenants)
- **Lift vs. random**
- **Revenue at Risk** (rental exposure × churn probability)

---

## 🧰 Replace Synthetic Data with Real Data
1. Export your **tenant‑month** dataset with at least these columns:
   - `tenant_id, property_id, ref_month, months_to_expiry, area_sqft, eff_rent_psf, rent_gap, late_pays_12m, tickets_12m, satisfaction, industry, submarket, label`
2. Save as CSV under `data/processed/tenant_month.csv`.
3. Adjust mapping/feature logic in `src/features.py` if needed.

---

## 🔍 Model Choices: Logistic Regression vs Gradient Boosting
- **Logistic Regression**: simple, interpretable baseline; good for early validation.
- **LightGBM**: usually higher predictive power; handles non-linearities & interactions; recommended for production ranking.

A recommended approach is to **start with Logistic Regression** for interpretability, then **confirm uplift** with LightGBM and deploy the better performing model.

---

## 📄 References (Background Reading)
- Savills / CBRE / JLL Hong Kong office market research for renewal & relocation dynamics (portfolio-level insights)
- Industry best practices on churn prediction (binary classification, Precision@K)

> This repository provides implementation patterns; please ensure compliance with data privacy and tenancy agreements when using real data.

