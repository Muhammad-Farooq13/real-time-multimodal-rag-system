from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BUNDLE_VERSION = "0.1.0"
MODELS_DIR = Path("models")
BUNDLE_PATH = MODELS_DIR / "demo_bundle.pkl"
RANDOM_SEED = 42


def generate_demo_data(n_rows: int = 900, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    contract_types = np.array(["monthly", "annual", "two_year"])
    regions = np.array(["north", "south", "east", "west", "central"])

    age = rng.integers(18, 76, size=n_rows)
    tenure_months = rng.integers(1, 73, size=n_rows)
    monthly_spend = rng.normal(loc=92, scale=24, size=n_rows).clip(18, 220)
    support_tickets = rng.poisson(lam=2.4, size=n_rows).clip(0, 12)
    usage_score = rng.normal(loc=67, scale=15, size=n_rows).clip(5, 100)
    satisfaction_score = rng.normal(loc=6.2, scale=1.8, size=n_rows).clip(1, 10)
    contract_type = rng.choice(contract_types, size=n_rows, p=[0.52, 0.31, 0.17])
    region = rng.choice(regions, size=n_rows)
    autopay = rng.choice(["yes", "no"], size=n_rows, p=[0.63, 0.37])
    paperless = rng.choice(["yes", "no"], size=n_rows, p=[0.74, 0.26])

    contract_risk = {"monthly": 0.85, "annual": -0.35, "two_year": -0.9}
    region_risk = {"north": 0.2, "south": 0.05, "east": -0.05, "west": 0.1, "central": -0.15}

    logits = (
        0.012 * (monthly_spend - 90)
        + 0.35 * support_tickets
        - 0.028 * tenure_months
        - 0.045 * (satisfaction_score - 5)
        - 0.018 * (usage_score - 60)
        + 0.012 * (35 - np.minimum(age, 35))
        + np.vectorize(contract_risk.get)(contract_type)
        + np.vectorize(region_risk.get)(region)
        + np.where(autopay == "no", 0.45, -0.1)
        + np.where(paperless == "yes", 0.12, -0.04)
        - 0.55
    )

    churn_probability = 1.0 / (1.0 + np.exp(-logits))
    churned = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend.round(2),
            "support_tickets": support_tickets,
            "usage_score": usage_score.round(2),
            "satisfaction_score": satisfaction_score.round(2),
            "contract_type": contract_type,
            "region": region,
            "autopay": autopay,
            "paperless": paperless,
            "churned": churned,
            "churn_probability": churn_probability.round(4),
        }
    )


def _feature_schema() -> dict:
    return {
        "numeric": {
            "age": {"min": 18, "max": 76, "default": 36},
            "tenure_months": {"min": 1, "max": 72, "default": 18},
            "monthly_spend": {"min": 18.0, "max": 220.0, "default": 95.0},
            "support_tickets": {"min": 0, "max": 12, "default": 2},
            "usage_score": {"min": 5.0, "max": 100.0, "default": 64.0},
            "satisfaction_score": {"min": 1.0, "max": 10.0, "default": 6.0},
        },
        "categorical": {
            "contract_type": ["monthly", "annual", "two_year"],
            "region": ["north", "south", "east", "west", "central"],
            "autopay": ["yes", "no"],
            "paperless": ["yes", "no"],
        },
    }


def _build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def train_and_build_bundle(output_path: Path = BUNDLE_PATH) -> dict:
    df = generate_demo_data()
    target = "churned"
    feature_columns = [column for column in df.columns if column not in {target, "churn_probability"}]
    numeric_features = [
        "age",
        "tenure_months",
        "monthly_spend",
        "support_tickets",
        "usage_score",
        "satisfaction_score",
    ]
    categorical_features = ["contract_type", "region", "autopay", "paperless"]

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_columns],
        df[target],
        test_size=0.22,
        stratify=df[target],
        random_state=RANDOM_SEED,
    )

    model_specs = {
        "Logistic Regression": LogisticRegression(max_iter=1200, solver="lbfgs"),
        "Random Forest": RandomForestClassifier(n_estimators=260, random_state=RANDOM_SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=320, random_state=RANDOM_SEED),
    }

    comparison_rows: list[dict] = []
    trained_models: dict[str, Pipeline] = {}
    best_name = ""
    best_auc = -1.0
    best_predictions: dict[str, np.ndarray] = {}

    for model_name, estimator in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", _build_preprocessor(numeric_features, categorical_features)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "model": model_name,
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
        }
        comparison_rows.append(metrics)
        trained_models[model_name] = pipeline

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = model_name
            best_predictions = {
                "y_true": y_test.to_numpy(),
                "y_pred": preds,
                "y_score": proba,
            }

    best_model = trained_models[best_name]
    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
    cm = confusion_matrix(best_predictions["y_true"], best_predictions["y_pred"]).tolist()

    model_step = best_model.named_steps["model"]
    preprocess = best_model.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out().tolist()
    importances = getattr(model_step, "feature_importances_", None)
    if importances is None:
        coefficients = getattr(model_step, "coef_", None)
        if coefficients is not None:
            importances = np.abs(coefficients[0])
    feature_importance_rows: list[dict] = []
    if importances is not None:
        top_idx = np.argsort(importances)[::-1][:10]
        feature_importance_rows = [
            {"feature": feature_names[int(idx)], "importance": round(float(importances[int(idx)]), 4)}
            for idx in top_idx
        ]

    analytics = {
        "target_rate": round(float(df[target].mean()), 4),
        "avg_monthly_spend": round(float(df["monthly_spend"].mean()), 2),
        "avg_tenure": round(float(df["tenure_months"].mean()), 2),
        "contract_mix": df.groupby("contract_type").size().rename("customers").reset_index().to_dict("records"),
        "regional_churn": (
            df.groupby("region")[target].mean().rename("churn_rate").reset_index().round(4).to_dict("records")
        ),
    }

    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "project_name": "Customer Churn Demo Dashboard",
        "target_name": target,
        "feature_columns": feature_columns,
        "feature_schema": _feature_schema(),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "models": trained_models,
        "best_model_name": best_name,
        "best_model": best_model,
        "model_results": comparison_df.to_dict("records"),
        "feature_importance": feature_importance_rows,
        "confusion_matrix": cm,
        "analytics": analytics,
        "full_dataframe": df.to_dict("records"),
        "sample_predictions": pd.DataFrame(
            {
                "actual": best_predictions["y_true"],
                "predicted": best_predictions["y_pred"],
                "score": np.round(best_predictions["y_score"], 4),
            }
        )
        .head(25)
        .to_dict("records"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return bundle


def load_demo_bundle(bundle_path: Path = BUNDLE_PATH) -> dict:
    with bundle_path.open("rb") as handle:
        bundle = pickle.load(handle)
    required_keys = {"best_model", "feature_schema", "model_results", "full_dataframe", "analytics"}
    missing = required_keys.difference(bundle.keys())
    if missing:
        raise ValueError(f"Bundle is missing required keys: {sorted(missing)}")
    return bundle


def load_or_rebuild_bundle(bundle_path: Path = BUNDLE_PATH) -> dict:
    try:
        return load_demo_bundle(bundle_path)
    except Exception:
        return train_and_build_bundle(bundle_path)


def main() -> None:
    bundle = train_and_build_bundle(BUNDLE_PATH)
    print(
        f"Built demo bundle at {BUNDLE_PATH} with best model {bundle['best_model_name']} "
        f"and ROC-AUC {bundle['model_results'][0]['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
