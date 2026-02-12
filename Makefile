
.PHONY: data logreg lgbm eval clean

VENV=.venv
PY=python

init:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -r requirements.txt

data:
	$(PY) src/generate_synthetic_data.py --n_tenants 1500 --months 24 --out data/processed/tenant_month.csv

logreg:
	$(PY) src/train.py --model logreg --config config/config.yaml --data data/processed/tenant_month.csv

lgbm:
	$(PY) src/train.py --model lgbm --config config/config.yaml --data data/processed/tenant_month.csv

eval:
	$(PY) src/evaluate.py --runs_dir runs --k 50 --revenue_col revenue_at_risk

clean:
	rm -rf runs/*
