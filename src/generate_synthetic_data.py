
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Synthetic data generator for tenant-month churn
# NOTE: Synthetic data is used strictly for portfolio demonstration purposes.

INDUSTRIES = [
    'Finance','Professional Services','Tech','Logistics','Media','Retail','Manufacturing','Healthcare'
]
SUBMARKETS = ['Central','Admiralty','Wanchai','Causeway Bay','Quarry Bay','Tsim Sha Tsui','Kowloon East','Others']


def generate_base_tenants(n_tenants:int, seed:int=2026):
    rng = np.random.default_rng(seed)
    tenants = []
    for i in range(n_tenants):
        tenant_id = f"T{i:05d}"
        property_id = f"P{rng.integers(1,200):04d}"
        industry = rng.choice(INDUSTRIES)
        submarket = rng.choice(SUBMARKETS, p=[0.18,0.08,0.09,0.08,0.20,0.12,0.20,0.05])
        area_sqft = max(300, int(rng.normal(8000, 4000)))
        eff_rent_psf = np.clip(rng.normal(55 if submarket!='Central' else 105, 15), 20, 160)
        lease_start_year = 2018 + int(rng.integers(0, 4))
        lease_term_years = rng.choice([3,4,5,6,7], p=[0.15,0.2,0.45,0.15,0.05])
        lease_start = pd.Timestamp(f"{lease_start_year}-{int(rng.integers(1,13))}-01")
        lease_expiry = lease_start + pd.DateOffset(years=int(lease_term_years))
        tenants.append({
            'tenant_id': tenant_id,
            'property_id': property_id,
            'industry': industry,
            'submarket': submarket,
            'area_sqft': area_sqft,
            'eff_rent_psf': float(eff_rent_psf),
            'lease_start': lease_start,
            'lease_expiry': lease_expiry
        })
    return pd.DataFrame(tenants)


def build_tenant_month_panel(tenants:pd.DataFrame, months:int=24, ref_last='2025-12-01', seed:int=2026):
    rng = np.random.default_rng(seed)
    ref_last = pd.Timestamp(ref_last)
    ref_months = pd.date_range(ref_last - pd.DateOffset(months=months-1), ref_last, freq='MS')

    rows = []
    for _, t in tenants.iterrows():
        # random behavior traits per tenant
        base_rent_gap = rng.normal(0.0, 0.08)  # % over/under market
        late_pay_rate = rng.uniform(0, 0.3)
        tickets_rate = rng.uniform(0.2, 1.2)
        satisfaction = np.clip(rng.normal(7.5, 1.5), 1, 10)

        for m in ref_months:
            if m < t['lease_start']:
                continue
            months_to_expiry = max(0, int((t['lease_expiry'] - m).days / 30.4))
            rent_gap = float(np.clip(base_rent_gap + rng.normal(0,0.03), -0.2, 0.3))
            late_pays_12m = int(np.clip(rng.poisson(lam=late_pay_rate*12), 0, 12))
            tickets_12m = int(np.clip(rng.poisson(lam=tickets_rate*6), 0, 30))
            # hazard components
            h_mte = 1 / (1 + np.exp(-(12 - months_to_expiry)/2))  # rises when within ~12 months
            h_gap = max(0, rent_gap)
            h_pay = late_pays_12m/12
            h_tix = tickets_12m/12
            h_sat = (10 - satisfaction)/10
            base_hazard = 0.02 + 0.25*h_mte + 0.25*h_gap + 0.2*h_pay + 0.15*h_tix + 0.1*h_sat
            # industry/submarket adjustments
            if t['industry'] in ['Tech','Finance']: base_hazard += 0.02
            if t['submarket'] in ['Kowloon East','Others']: base_hazard += 0.015
            prob_churn_12m = float(np.clip(base_hazard, 0, 0.95))
            label = int(rng.random() < prob_churn_12m)

            # approximate revenue at risk (12 months exposure for simplicity)
            rev_at_risk = t['area_sqft'] * t['eff_rent_psf'] * 12

            rows.append({
                'tenant_id': t['tenant_id'],
                'property_id': t['property_id'],
                'ref_month': m,
                'months_to_expiry': months_to_expiry,
                'industry': t['industry'],
                'submarket': t['submarket'],
                'area_sqft': t['area_sqft'],
                'eff_rent_psf': t['eff_rent_psf'],
                'rent_gap': rent_gap,
                'late_pays_12m': late_pays_12m,
                'tickets_12m': tickets_12m,
                'satisfaction': satisfaction,
                'lease_start': t['lease_start'],
                'lease_expiry': t['lease_expiry'],
                'revenue_at_risk': rev_at_risk,
                'label': label
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_tenants', type=int, default=1500)
    ap.add_argument('--months', type=int, default=24)
    ap.add_argument('--out', type=str, default='data/processed/tenant_month.csv')
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    tenants = generate_base_tenants(args.n_tenants, seed=args.seed)
    panel = build_tenant_month_panel(tenants, months=args.months, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)
    print(f"Saved synthetic dataset to {out_path} with shape {panel.shape}")

if __name__ == '__main__':
    main()
