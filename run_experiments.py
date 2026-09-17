"""Compare candidate regression models with 5-fold CV + hold-out test.

Usage:  python scripts/run_experiments.py

"""
from model import get_model_data
from model import EXPERIMENTS_PATH, compare_models


def main() -> None:
    df = get_model_data()
    print(f"Training on {len(df):,} clean listings\n")
    results = compare_models(df)
    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(EXPERIMENTS_PATH, index=False)
    with_fmt = results.copy()
    for c in ["MAE (₹)", "RMSE (₹)"]:
        with_fmt[c] = with_fmt[c].map(lambda v: f"{v:,.0f}")
    print(with_fmt.round(3).to_string(index=False))
    print(f"\nSaved -> {EXPERIMENTS_PATH}")


if __name__ == "__main__":
    main()
