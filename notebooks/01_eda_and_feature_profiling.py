
# %% [markdown]
# # EDA & Feature Profiling (Tenant-Month Dataset)
# This script can be opened as a notebook via Jupytext, or run as a Python script.

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path('data/processed/tenant_month.csv')
df = pd.read_csv(DATA_PATH, parse_dates=['ref_month', 'lease_start', 'lease_expiry'])
print(df.head())
print(df.describe(include='all'))

# %%
# Correlation heatmap for numeric features
num_cols = ['months_to_expiry','area_sqft','eff_rent_psf','rent_gap','late_pays_12m','tickets_12m','satisfaction']
plt.figure(figsize=(10,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='Blues')
plt.title('Numeric Feature Correlations')
plt.tight_layout()
plt.show()

# %%
# Churn rate by months_to_expiry bins
bins = [-1,3,6,9,12,24]
df['mte_bin'] = pd.cut(df['months_to_expiry'], bins)
print(df.groupby('mte_bin')['label'].mean().rename('churn_rate'))

# %%
# Churn by industry
print(df.groupby('industry')['label'].mean().sort_values(ascending=False).head(10))
