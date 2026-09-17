"""
AutoIQ - Price Prediction Model
===============================
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, get_clean_data, prepare_for_model, load_raw

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "autoiq_price_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
EXPERIMENTS_PATH = Path(__file__).resolve().parent / "model_comparison.csv"

RANDOM_STATE = 42
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_model_data(force_rebuild: bool = False) -> pd.DataFrame:
    """Return the dataset used for MODEL training: parsed, de-duplicated and
    outlier-filtered, but WITH missing values left in place so the pipeline
    imputer can handle them on the training split only (no leakage).

    Cached to model_data.csv alongside the module."""
    cache = MODEL_DIR / "model_data.csv"
    if cache.exists() and not force_rebuild:
        return pd.read_csv(cache)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_for_model(load_raw())
    df.to_csv(cache, index=False)
    return df


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #

def build_preprocessor() -> ColumnTransformer:
    """Impute, then scale numeric columns; impute, then one-hot encode
    categoricals.

    Imputation is the FIRST step inside each branch so it is fitted on the
    training fold only (during cross-validation and the final fit). This is
    what prevents test-set leakage: median/most-frequent values are learned
    from training data alone, never from the held-out rows.

    `handle_unknown='ignore'` means a brand unseen at training time (e.g. a
    dealer uploads a Porsche listing) is encoded as all-zeros instead of
    raising an error - important for the upload feature.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict[str, object]:
    """Model families compared in the experiment. Hyper-parameters are
    deliberately modest so the comparison reflects model *type*, not tuning."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.9, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #

def _metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    """Metrics are reported in rupees (back-transformed) so stakeholders
    can interpret them, plus R² on the log scale used for fitting."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "MAE (₹)": float(mean_absolute_error(y_true, y_pred)),
        "RMSE (₹)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE (%)": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "R² (log)": float(r2_score(y_true_log, y_pred_log)),
        "R² (₹)": float(r2_score(y_true, y_pred)),
    }


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    X = df[FEATURES]
    y = np.log1p(df[TARGET])
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def compare_models(df: pd.DataFrame, cv_folds: int = 5) -> pd.DataFrame:
    """Cross-validate every candidate on the training split and evaluate on a
    held-out test split. Returns a tidy comparison table."""
    X_train, X_test, y_train, y_test = split_data(df)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, est in candidate_models().items():
        pipe = build_pipeline(est)
        cv = cross_validate(pipe, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
        pipe.fit(X_train, y_train)
        test_metrics = _metrics(y_test.to_numpy(), pipe.predict(X_test))
        rows.append({
            "Model": name,
            "CV R² mean": float(cv["test_score"].mean()),
            "CV R² std": float(cv["test_score"].std()),
            **test_metrics,
            "Fit time (s)": float(cv["fit_time"].mean()),
        })
    return pd.DataFrame(rows).sort_values("MAE (₹)").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Training / persistence
# --------------------------------------------------------------------------- #

def train_final_model(df: pd.DataFrame, estimator_name: str = "Gradient Boosting") -> tuple[Pipeline, dict]:
   
    X_train, X_test, y_train, y_test = split_data(df)
    pipe = build_pipeline(candidate_models()[estimator_name])
    pipe.fit(X_train, y_train)
    metrics = _metrics(y_test.to_numpy(), pipe.predict(X_test))
    metrics.update({
        "model": estimator_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
    })
    # Keep test-set predictions for the Actual vs Predicted chart
    metrics["_test_actual"] = np.expm1(y_test).round(0).tolist()
    metrics["_test_pred"] = np.expm1(pipe.predict(X_test)).round(0).tolist()

    pipe.fit(df[FEATURES], np.log1p(df[TARGET]))   # refit on everything
    return pipe, metrics


def save_model(pipe: Pipeline, metrics: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def load_model() -> tuple[Pipeline, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained yet. Run: python scripts/train_model.py")
    return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text())


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def predict_price(pipe: Pipeline, car: dict, mape: float | None = None) -> dict:
    """Estimate a price for one car from historical data. Returns a point
    estimate plus an INDICATIVE price range (not a statistical confidence
    interval): the range is simply the estimate +/- the model's test-set
    MAPE, giving a rough sense of typical error, and is labelled as such
    in the UI."""
    X = pd.DataFrame([car])[FEATURES]
    price = float(np.expm1(pipe.predict(X)[0]))
    band = (mape or 15.0) / 100
    return {
        "predicted_price": round(price, -2),
        "low": round(price * (1 - band), -2),
        "high": round(price * (1 + band), -2),
    }


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    """Aggregate one-hot importances back to their parent feature so the
    chart shows 'brand' once rather than 30 brand columns."""
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    names = pipe.named_steps["prep"].get_feature_names_out()
    raw = pd.Series(model.feature_importances_, index=names)
    # categorical names come back as cat__brand_Maruti; map to 'brand'
    parent = []
    for n in names:
        col = n.split("__", 1)[1]
        for c in CATEGORICAL_FEATURES:
            if col.startswith(c + "_"):
                col = c
                break
        parent.append(col)
    agg = raw.groupby(parent).sum().sort_values(ascending=False)
    return agg.reset_index().rename(columns={"index": "feature", 0: "importance"})
