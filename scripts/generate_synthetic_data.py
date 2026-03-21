"""Generate a synthetic telecom customer dataset for propensity modelling."""

import pathlib
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_CUSTOMERS = 600
OUTPUT_PATH = pathlib.Path("data/raw/telecom_customers.csv")


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)

    # ── Demographics ──────────────────────────────────────────────────────────
    age = rng.integers(18, 76, N_CUSTOMERS)
    tenure_months = rng.integers(1, 121, N_CUSTOMERS)
    region = rng.choice(["North", "South", "East", "West"], N_CUSTOMERS)

    # ── Current plan / usage ──────────────────────────────────────────────────
    contract_type = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        N_CUSTOMERS,
        p=[0.55, 0.30, 0.15],
    )
    internet_service = rng.choice(
        ["Fiber", "DSL", "None"], N_CUSTOMERS, p=[0.45, 0.40, 0.15]
    )
    monthly_charges = rng.uniform(20, 100, N_CUSTOMERS).round(2)
    data_usage_gb = rng.exponential(scale=8, size=N_CUSTOMERS).clip(0, 50).round(2)
    call_minutes = rng.integers(0, 1001, N_CUSTOMERS)

    # ── Customer behaviour ────────────────────────────────────────────────────
    num_products = rng.integers(1, 5, N_CUSTOMERS)
    has_streaming = rng.choice([0, 1], N_CUSTOMERS, p=[0.4, 0.6])
    has_device_protection = rng.choice([0, 1], N_CUSTOMERS, p=[0.6, 0.4])
    num_complaints = rng.integers(0, 6, N_CUSTOMERS)
    customer_service_calls = rng.integers(0, 11, N_CUSTOMERS)

    # ── Propensity score (ground-truth signal) ────────────────────────────────
    # Higher data usage, longer tenure, fiber internet → more likely to upgrade
    # High charges, many complaints, many products (already engaged) → less likely
    logit = (
        -2.0
        + 0.04 * data_usage_gb
        + 0.008 * tenure_months
        + 0.5 * (internet_service == "Fiber").astype(float)
        - 0.01 * monthly_charges
        - 0.2 * num_complaints
        - 0.1 * (contract_type == "two-year").astype(float)
        - 0.05 * num_products
        + rng.normal(0, 0.3, N_CUSTOMERS)
    )
    prob_upgrade = 1 / (1 + np.exp(-logit))
    upgraded = (rng.uniform(0, 1, N_CUSTOMERS) < prob_upgrade).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:04d}" for i in range(N_CUSTOMERS)],
            "age": age,
            "tenure_months": tenure_months,
            "region": region,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "monthly_charges": monthly_charges,
            "data_usage_gb": data_usage_gb,
            "call_minutes": call_minutes,
            "num_products": num_products,
            "has_streaming": has_streaming,
            "has_device_protection": has_device_protection,
            "num_complaints": num_complaints,
            "customer_service_calls": customer_service_calls,
            "upgraded": upgraded,
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    upgrade_rate = upgraded.mean()
    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")
    print(f"Upgrade rate: {upgrade_rate:.1%}  ({upgraded.sum()} / {len(df)})")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
