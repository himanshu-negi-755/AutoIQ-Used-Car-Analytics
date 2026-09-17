"""Train the production price model and persist it with its metrics.

Usage:  python scripts/train_model.py [--model "Random Forest"]
"""
import argparse
from model import get_model_data
from model import MODEL_PATH, candidate_models, save_model, train_final_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Gradient Boosting", choices=list(candidate_models()))
    args = parser.parse_args()

    df = get_model_data()
    pipe, metrics = train_final_model(df, args.model)
    save_model(pipe, metrics)

    print(f"Model: {metrics['model']}  |  train={metrics['n_train']:,}  test={metrics['n_test']:,}")
    for k in ["MAE (₹)", "RMSE (₹)", "MAPE (%)", "R² (log)", "R² (₹)"]:
        print(f"  {k:<10} {metrics[k]:,.3f}")
    print(f"Saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
