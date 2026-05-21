"""
evaluate.py — TSTR evaluation for Binary Diffusion synthetic tabular data.

Protocol (from paper):
  - 5 synthetic training sets generated from a model trained on real data.
  - Each set is used to train LR, DT, RF; evaluated on the real test set.
  - Report mean ± std of accuracy (classification) or MSE (regression).

Usage:
    python evaluate.py --config config.yaml
"""

import argparse, yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree         import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble     import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics      import accuracy_score, mean_squared_error


# Hyperparameters per dataset (paper, Table 2).
# California Housing is regression → LR = LinearRegression (lr_max_iter unused).
HYPERPARAMS = {
    "travel":     dict(lr_max_iter=100,  dt_max_depth=6,  rf_max_depth=12, rf_n_estimators=75),
    "sick":       dict(lr_max_iter=200,  dt_max_depth=10, rf_max_depth=12, rf_n_estimators=90),
    "heloc":      dict(lr_max_iter=500,  dt_max_depth=6,  rf_max_depth=12, rf_n_estimators=78),
    "adult":      dict(lr_max_iter=1000, dt_max_depth=8,  rf_max_depth=12, rf_n_estimators=85),
    "diabetes":   dict(lr_max_iter=500,  dt_max_depth=10, rf_max_depth=20, rf_n_estimators=120),
    "california": dict(lr_max_iter=None, dt_max_depth=10, rf_max_depth=12, rf_n_estimators=85),
}


def load(path, drop_cols):
    """Load CSV, replace '?' with NaN, drop unwanted columns."""
    df = pd.read_csv(path, na_values=["?"])
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def fill_na(df):
    """Fill NaN: mean for numeric columns, mode for string/object columns."""
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype == object:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
    return df


def to_numpy(X_syn, X_test, y_syn, y_test, task):
    """
    Convert DataFrames to numpy arrays ready for sklearn — no external encoders needed.

    String/object columns are converted to integer category codes using pandas.
    Categories are always derived from the REAL test data so that both synthetic
    train and real test share the same consistent mapping. Unknown synthetic
    values (not seen in real test) are assigned code -1, which is harmless.
    Numeric columns (int, float) are passed through as-is.
    """
    X_syn, X_test = X_syn.copy(), X_test.copy()

    # Auto-detect and encode string columns (no need to declare them in config)
    for col in X_test.select_dtypes(include="object").columns:
        cat_type = pd.CategoricalDtype(categories=X_test[col].astype(str).unique())
        X_test[col] = X_test[col].astype(str).astype(cat_type).cat.codes
        X_syn[col]  = X_syn[col].astype(str).astype(cat_type).cat.codes

    # Encode target for classification the same way
    if task == "classification" and y_test.dtype == object:
        cat_type = pd.CategoricalDtype(categories=y_test.astype(str).unique())
        y_test = y_test.astype(str).astype(cat_type).cat.codes
        y_syn  = y_syn.astype(str).astype(cat_type).cat.codes

    return (X_syn.values.astype(float), X_test.values.astype(float),
            y_syn.values.astype(float),  y_test.values.astype(float))


def build_models(task, hp):
    """Return LR, DT, RF with dataset-specific hyperparameters."""
    if task == "classification":
        return {
            "LR": LogisticRegression(max_iter=hp["lr_max_iter"], n_jobs=-1, random_state=42),
            "DT": DecisionTreeClassifier(max_depth=hp["dt_max_depth"], random_state=42),
            "RF": RandomForestClassifier(max_depth=hp["rf_max_depth"],
                                         n_estimators=hp["rf_n_estimators"],
                                         n_jobs=-1, random_state=42),
        }
    else:
        return {
            "LR": LinearRegression(n_jobs=-1),
            "DT": DecisionTreeRegressor(max_depth=hp["dt_max_depth"], random_state=42),
            "RF": RandomForestRegressor(max_depth=hp["rf_max_depth"],
                                        n_estimators=hp["rf_n_estimators"],
                                        n_jobs=-1, random_state=42),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg      = yaml.safe_load(open(parser.parse_args().config))
    data_cfg = cfg["data"]
    dataset  = cfg["dataset_name"].lower()
    task     = data_cfg["task"]            # "classification" or "regression"
    target   = data_cfg["target_column"]
    drop     = data_cfg.get("columns_to_drop") or []

    assert dataset in HYPERPARAMS, f"Unknown dataset '{dataset}'. Choose from: {list(HYPERPARAMS)}"

    # Load and clean the real test set (fixed reference across all 5 runs)
    test_df    = fill_na(load(cfg["path_test"], drop))
    X_test_raw = test_df.drop(columns=[target])
    y_test_raw = test_df[target]

    # Accumulate scores over the 5 synthetic training sets
    scores = {name: [] for name in ["LR", "DT", "RF"]}
    metric_fn = accuracy_score if task == "classification" else mean_squared_error

    for i, syn_path in enumerate(cfg["path_synthetic_trains"], 1):
        print(f"  [run {i}/{len(cfg['path_synthetic_trains'])}] {syn_path}")

        syn_df    = fill_na(load(syn_path, drop))
        X_syn_raw = syn_df.drop(columns=[target])
        y_syn_raw = syn_df[target]

        # Convert to numpy (string columns encoded via real test categories)
        X_syn, X_test, y_syn, y_test = to_numpy(
            X_syn_raw, X_test_raw, y_syn_raw, y_test_raw, task
        )

        # Train on synthetic, score on real — the TSTR protocol
        for name, model in build_models(task, HYPERPARAMS[dataset]).items():
            model.fit(X_syn, y_syn)
            scores[name].append(metric_fn(y_test, model.predict(X_test)))

    # Results table
    metric_label = "Accuracy ↑" if task == "classification" else "MSE ↓"
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.upper()} | Task: {task} | Metric: {metric_label}")
    print(f"{'='*60}")
    print(f"{'Model':<8}  {'Mean':>10}  {'Std':>10}  All scores")
    print(f"{'-'*60}")
    for name, s in scores.items():
        print(f"{name:<8}  {np.mean(s):>10.4f}  {np.std(s):>10.4f}  "
              f"[{'  '.join(f'{x:.4f}' for x in s)}]")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
