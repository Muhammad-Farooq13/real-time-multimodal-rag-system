from __future__ import annotations

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BUNDLE_VERSION = "0.1.0"
MODELS_DIR = Path("models")
BUNDLE_PATH = MODELS_DIR / "demo_bundle.pkl"
RANDOM_SEED = 42


class PortableChurnModel:
    def __init__(self) -> None:
        self.contract_risk = {"monthly": 0.85, "annual": -0.35, "two_year": -0.9}
        self.region_risk = {"north": 0.2, "south": 0.05, "east": -0.05, "west": 0.1, "central": -0.15}

    def _score(self, features: pd.DataFrame) -> np.ndarray:
        frame = features.copy()
        logits = (
            0.012 * (pd.to_numeric(frame["monthly_spend"], errors="coerce").fillna(95.0) - 90)
            + 0.35 * pd.to_numeric(frame["support_tickets"], errors="coerce").fillna(2.0)
            - 0.028 * pd.to_numeric(frame["tenure_months"], errors="coerce").fillna(18.0)
            - 0.045 * (pd.to_numeric(frame["satisfaction_score"], errors="coerce").fillna(6.0) - 5)
            - 0.018 * (pd.to_numeric(frame["usage_score"], errors="coerce").fillna(64.0) - 60)
            + 0.012 * (35 - np.minimum(pd.to_numeric(frame["age"], errors="coerce").fillna(36.0), 35))
            + frame["contract_type"].map(self.contract_risk).fillna(0.0)
            + frame["region"].map(self.region_risk).fillna(0.0)
            + np.where(frame["autopay"].fillna("yes") == "no", 0.45, -0.1)
            + np.where(frame["paperless"].fillna("yes") == "yes", 0.12, -0.04)
            - 0.55
        )
        return 1.0 / (1.0 + np.exp(-logits.to_numpy(dtype=float)))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.clip(self._score(features), 1e-4, 1 - 1e-4)
        return np.column_stack([1.0 - probabilities, probabilities])


if __name__ == "__main__":
    sys.modules.setdefault("train_demo", sys.modules[__name__])

PortableChurnModel.__module__ = "train_demo"


def _compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = max(len(y_true), 1)
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    roc_auc = _compute_roc_auc(y_true, y_score)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
    }


def _compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5

    wins = 0.0
    for positive in positives:
        wins += float(np.sum(positive > negatives))
        wins += 0.5 * float(np.sum(positive == negatives))
    return wins / (len(positives) * len(negatives))


def _train_test_split_portable(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.22,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_state)
    positive_idx = np.flatnonzero(target.to_numpy() == 1)
    negative_idx = np.flatnonzero(target.to_numpy() == 0)
    rng.shuffle(positive_idx)
    rng.shuffle(negative_idx)

    positive_test = max(1, int(round(len(positive_idx) * test_size)))
    negative_test = max(1, int(round(len(negative_idx) * test_size)))
    test_idx = np.concatenate([positive_idx[:positive_test], negative_idx[:negative_test]])
    train_idx = np.concatenate([positive_idx[positive_test:], negative_idx[negative_test:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    return (
        features.iloc[train_idx].reset_index(drop=True),
        features.iloc[test_idx].reset_index(drop=True),
        target.iloc[train_idx].reset_index(drop=True),
        target.iloc[test_idx].reset_index(drop=True),
    )


def _load_sklearn() -> dict[str, Any]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return {
        "ColumnTransformer": ColumnTransformer,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


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


def _build_preprocessor(numeric_features: list[str], categorical_features: list[str], sklearn_modules: dict[str, Any]) -> Any:
    pipeline_cls = sklearn_modules["Pipeline"]
    simple_imputer = sklearn_modules["SimpleImputer"]
    standard_scaler = sklearn_modules["StandardScaler"]
    one_hot_encoder = sklearn_modules["OneHotEncoder"]
    column_transformer = sklearn_modules["ColumnTransformer"]

    numeric_transformer = pipeline_cls(
        steps=[("imputer", simple_imputer(strategy="median")), ("scaler", standard_scaler())]
    )
    categorical_transformer = pipeline_cls(
        steps=[
            ("imputer", simple_imputer(strategy="most_frequent")),
            ("encoder", one_hot_encoder(handle_unknown="ignore")),
        ]
    )
    return column_transformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def _build_portable_bundle(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    comparison_rows: list[dict[str, Any]],
    feature_importance_rows: list[dict[str, Any]],
    best_name: str,
) -> dict:
    portable_model = PortableChurnModel()
    y_score = portable_model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    confusion = [
        [int(np.sum((y_test.to_numpy() == 0) & (y_pred == 0))), int(np.sum((y_test.to_numpy() == 0) & (y_pred == 1)))],
        [int(np.sum((y_test.to_numpy() == 1) & (y_pred == 0))), int(np.sum((y_test.to_numpy() == 1) & (y_pred == 1)))],
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

    sample_predictions = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": y_pred,
            "score": np.round(y_score, 4),
        }
    )

    return {
        "bundle_version": BUNDLE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "project_name": "Customer Churn Demo Dashboard",
        "target_name": target,
        "feature_columns": feature_columns,
        "feature_schema": _feature_schema(),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "best_model_name": best_name,
        "best_model": portable_model,
        "model_results": comparison_rows,
        "feature_importance": feature_importance_rows,
        "confusion_matrix": confusion,
        "analytics": analytics,
        "full_dataframe": df.to_dict("records"),
        "sample_predictions": sample_predictions.head(25).to_dict("records"),
    }


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

    X_train, X_test, y_train, y_test = _train_test_split_portable(
        df[feature_columns],
        df[target],
    )

    portable_model = PortableChurnModel()
    portable_scores = portable_model.predict_proba(X_test)[:, 1]
    comparison_rows = [{"model": "Portable Heuristic", **_compute_binary_metrics(y_test.to_numpy(), portable_scores)}]
    feature_importance_rows = [
        {"feature": "contract_type_monthly", "importance": 0.85},
        {"feature": "support_tickets", "importance": 0.35},
        {"feature": "autopay_no", "importance": 0.45},
        {"feature": "tenure_months", "importance": 0.028},
        {"feature": "satisfaction_score", "importance": 0.045},
        {"feature": "usage_score", "importance": 0.018},
    ]
    best_name = "Portable Heuristic"

    try:
        sklearn_modules = _load_sklearn()
        train_test_split = sklearn_modules["train_test_split"]
        logistic_regression = sklearn_modules["LogisticRegression"]
        random_forest = sklearn_modules["RandomForestClassifier"]
        extra_trees = sklearn_modules["ExtraTreesClassifier"]
        pipeline_cls = sklearn_modules["Pipeline"]
        accuracy_score = sklearn_modules["accuracy_score"]
        precision_score = sklearn_modules["precision_score"]
        recall_score = sklearn_modules["recall_score"]
        f1_score = sklearn_modules["f1_score"]
        roc_auc_score = sklearn_modules["roc_auc_score"]

        X_train, X_test, y_train, y_test = train_test_split(
            df[feature_columns],
            df[target],
            test_size=0.22,
            stratify=df[target],
            random_state=RANDOM_SEED,
        )

        model_specs = {
            "Logistic Regression": logistic_regression(max_iter=1200, solver="lbfgs"),
            "Random Forest": random_forest(n_estimators=260, random_state=RANDOM_SEED),
            "Extra Trees": extra_trees(n_estimators=320, random_state=RANDOM_SEED),
        }
        trained_feature_importance: list[dict[str, Any]] = []
        best_auc = -1.0

        comparison_rows = []
        for model_name, estimator in model_specs.items():
            pipeline = pipeline_cls(
                steps=[
                    ("preprocess", _build_preprocessor(numeric_features, categorical_features, sklearn_modules)),
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

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_name = model_name
                model_step = pipeline.named_steps["model"]
                preprocess = pipeline.named_steps["preprocess"]
                feature_names = preprocess.get_feature_names_out().tolist()
                importances = getattr(model_step, "feature_importances_", None)
                if importances is None:
                    coefficients = getattr(model_step, "coef_", None)
                    if coefficients is not None:
                        importances = np.abs(coefficients[0])
                if importances is not None:
                    top_idx = np.argsort(importances)[::-1][:10]
                    trained_feature_importance = [
                        {"feature": feature_names[int(idx)], "importance": round(float(importances[int(idx)]), 4)}
                        for idx in top_idx
                    ]

        comparison_rows = pd.DataFrame(comparison_rows).sort_values(by="roc_auc", ascending=False).to_dict("records")
        if trained_feature_importance:
            feature_importance_rows = trained_feature_importance
    except ModuleNotFoundError:
        pass

    bundle = _build_portable_bundle(
        df=df,
        feature_columns=feature_columns,
        target=target,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        comparison_rows=comparison_rows,
        feature_importance_rows=feature_importance_rows,
        best_name=best_name,
    )

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
    if not hasattr(bundle["best_model"], "predict_proba"):
        raise ValueError("Bundle best_model does not support predict_proba")
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
