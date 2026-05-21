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
from typing import Optional
from importlib import metadata
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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


def load(path: str):
    """Load CSV, replace '?' with NaN, drop unwanted columns."""
    df = pd.read_csv(path, na_values=["?"])
    return df


def preprocess(df: pd.DataFrame, cfg: dict, target: str, metadata: dict):
    """
    Fill NaN, binarize features, return (X_binary, y, metadata).

    The `metadata` dict is populated on the first call (synthetic training set)
    and reused as-is on subsequent calls (test set), ensuring that:
      - numerical columns use the same [min, max] range for bit-encoding
      - categorical columns use the same category→index mapping and bit width
    This guarantees every row in both datasets has exactly the same binary length.
    """

    col_drop        = cfg["data"].get("columns_to_drop")    or []
    col_categorical = cfg["data"].get("categorical_columns") or []
    col_numerical   = cfg["data"].get("numerical_columns")   or []

    col_drop        = [c for c in col_drop        if c in df.columns]
    col_categorical = [c for c in col_categorical if c in df.columns]
    col_numerical   = [c for c in col_numerical   if c in df.columns]

    df = df.drop(columns=col_drop)

    for col in col_categorical:
        df[col] = df[col].astype("category")
    for col in col_numerical:
        df[col] = df[col].astype("float64")

    if cfg["data"].get("dropna", False):
        df = df.dropna()
    else:
        for col in col_numerical:
            df[col] = df[col].fillna(df[col].mean())
        for col in col_categorical:
            df[col] = df[col].fillna(df[col].mode()[0])

    if cfg["dataset_name"].lower() == "california":
        df["median_house_value"] = df["median_house_value"] / 100_000.0

    X = df.drop(columns=[target]).copy()
    y = df[target]

    # ------------------------------------------------------------------
    # Numerical columns → 32-bit binary string
    # Min/max are recorded from the first (synthetic) call and reused for
    # the test set so the encoding range is always identical.
    # Values are clamped to [0, 1] before encoding to handle out-of-range
    # test values without producing negative integers or overflow.
    # ------------------------------------------------------------------
    def numerical_to_binary(val: float, min_val: float, max_val: float) -> str:
        size = 32
        if max_val == min_val:          # constant column — all zeros
            return "0" * size
        normalized = (val - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))   # clamp → no '-' prefix
        return format(int(normalized * (2 ** size - 1)), f"0{size}b")

    for col in col_numerical:
        if col not in metadata:
            metadata[col] = {
                "min": float(X[col].min()),
                "max": float(X[col].max()),
            }
        min_val = metadata[col]["min"]
        max_val = metadata[col]["max"]
        # .astype(object) strips the category dtype before applying
        X[col] = (
            X[col]
            .astype(object)
            .apply(lambda x: numerical_to_binary(x, min_val, max_val))
        )

    # ------------------------------------------------------------------
    # Categorical columns → fixed-width binary string
    # The category map and bit width come from the first (synthetic) call.
    # Unknown test categories are mapped to index 0 (a safe default).
    # ------------------------------------------------------------------
    for col in col_categorical:
        if col not in metadata:
            metadata[col] = {
                "category_map": {
                    cat: idx for idx, cat in enumerate(X[col].unique())
                }
            }
        category_map  = metadata[col]["category_map"]
        unique_values = len(category_map)
        size = int(np.ceil(np.log2(unique_values))) if unique_values > 1 else 1
        # .astype(object) strips the category dtype so .apply behaves predictably
        X[col] = (
            X[col]
            .astype(object)
            .apply(lambda x: format(category_map.get(x, 0), f"0{size}b"))
        )

    # ------------------------------------------------------------------
    # Join all binary strings per row → 1-D int array
    # All columns are now plain Python strings, so "".join is safe.
    # ------------------------------------------------------------------
    X_str = X.astype(str)   # uniform object→str, no category weirdness

    rows_binary = []
    for i in range(len(X_str)):
        row_str = "".join(X_str.iloc[i].values)
        if any(c not in ("0", "1") for c in row_str):
            raise ValueError(
                f"Row {i} contains non-binary characters: "
                f"{set(row_str) - {'0','1'}}. "
                "Likely a NaN was not filled — check your config."
            )
        rows_binary.append(np.frombuffer(row_str.encode(), dtype=np.uint8) - ord("0"))

    lengths = {len(r) for r in rows_binary}
    if len(lengths) > 1:
        raise ValueError(
            f"Rows have inconsistent binary lengths {lengths}. "
            "This means different rows produced different total bit counts — "
            "check for unseen categories or NaN leakage."
        )

    X_binary = np.stack(rows_binary).astype(np.int8)
    return X_binary, y, metadata


def build_models(task: str, hp: dict, seed: int = 0):
    """Return LR, DT, RF with dataset-specific hyperparameters."""
    if task == "classification":
        return {
            "LR": LogisticRegression(max_iter=hp["lr_max_iter"], n_jobs=-1, random_state=seed),
            "DT": DecisionTreeClassifier(max_depth=hp["dt_max_depth"], random_state=seed),
            "RF": RandomForestClassifier(
                max_depth=hp["rf_max_depth"],
                n_estimators=hp["rf_n_estimators"],
                n_jobs=-1, random_state=seed,
            ),
        }
    else:
        return {
            "LR": LinearRegression(n_jobs=-1),
            "DT": DecisionTreeRegressor(max_depth=hp["dt_max_depth"], random_state=seed),
            "RF": RandomForestRegressor(
                max_depth=hp["rf_max_depth"],
                n_estimators=hp["rf_n_estimators"],
                n_jobs=-1, random_state=seed,
            ),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg      = yaml.safe_load(open(parser.parse_args().config))
    data_cfg = cfg["data"]
    dataset  = cfg["dataset_name"].lower()
    task     = data_cfg["task"]          # "classification" or "regression"
    target   = data_cfg["target_column"]

    assert dataset in HYPERPARAMS, (
        f"Unknown dataset '{dataset}'. Choose from: {list(HYPERPARAMS)}"
    )

    scores    = {name: [] for name in ["LR", "DT", "RF"]}
    metric_fn = accuracy_score if task == "classification" else mean_squared_error

    for i, syn_path in enumerate(cfg["path_synthetic_trains"], 1):
        print(f"  [run {i}/{len(cfg['path_synthetic_trains'])}] {syn_path}")

        # ── TSTR order ────────────────────────────────────────────────
        # 1. Fit metadata (min/max, category maps) on synthetic data.
        # 2. Apply that same metadata when encoding the test set so both
        #    datasets share identical column ranges and bit widths.
        # ──────────────────────────────────────────────────────────────
        X_synthetic, y_synthetic, syn_metadata = preprocess(
            load(syn_path), cfg, target, metadata={}
        )
        X_test, y_test, _ = preprocess(
            load(cfg["path_test"]), cfg, target, metadata=syn_metadata
        )

        if task == "classification":
            le = LabelEncoder()
            # Fit on the union of labels so both sets use the same encoding
            all_labels = np.concatenate([y_synthetic.values, y_test.values])
            le.fit(all_labels)
            y_synthetic_enc = le.transform(y_synthetic.values)
            y_test_enc      = le.transform(y_test.values)
        else:
            scaler = MinMaxScaler()
            y_synthetic_enc = scaler.fit_transform(
                y_synthetic.values.reshape(-1, 1)
            ).flatten()
            # y_test stays in original scale; predictions are inverse-transformed
            y_test_enc = y_test.values

        for name, model in build_models(task, HYPERPARAMS[dataset]).items():
            model.fit(X_synthetic, y_synthetic_enc)
            y_pred = model.predict(X_test)

            if task == "regression":
                y_pred = scaler.inverse_transform(
                    y_pred.reshape(-1, 1)
                ).flatten()

            scores[name].append(metric_fn(y_test_enc, y_pred))

    # Results table
    metric_label = "Accuracy" if task == "classification" else "MSE"
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.upper()} | Task: {task} | Metric: {metric_label}")
    print(f"{'='*60}")
    print(f"{'Model':<8}  {'Mean':>10}  {'Std':>10}  All scores")
    print(f"{'-'*60}")
    for name, s in scores.items():
        print(
            f"{name:<8}  {np.mean(s):>10.4f}  {np.std(s):>10.4f}  "
            f"[{'  '.join(f'{x:.4f}' for x in s)}]"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
