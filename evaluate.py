"""
evaluate.py — TSTR evaluation for Binary Diffusion synthetic tabular data.

Protocol (from paper):
  - 5 synthetic training sets generated from a model trained on real data.
  - Each set is used to train LR, DT, RF; evaluated on the real test set.
  - Report mean ± std of accuracy (classification) or MSE (regression).

CLI usage:
    python evaluate.py --config config.yaml

Notebook usage:
    from evaluate import evaluate
    import yaml

    cfg     = yaml.safe_load(open("eval_configs/diabetes.yml"))
    results = evaluate(cfg)
    # results = {
    #   "dataset": "diabetes",
    #   "task": "classification",
    #   "metric": "accuracy",
    #   "models": {
    #     "LR": {"mean": 0.82, "std": 0.01, "scores": [...]},
    #     "DT": {...},
    #     "RF": {...},
    #   }
    # }
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model  import LogisticRegression, LinearRegression
from sklearn.tree          import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble      import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics       import accuracy_score, mean_squared_error


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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    """Load CSV, replace '?' with NaN."""
    return pd.read_csv(path, na_values=["?"])


# ---------------------------------------------------------------------------
# Preprocessing / binarisation
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, cfg: dict, target: str, metadata: dict):
    """
    Fill NaN, binarize features, return (X_binary, y, metadata).

    The `metadata` dict is populated on the first call (synthetic training set)
    and reused as-is on subsequent calls (test set), ensuring:
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

    col_categorical = [c for c in col_categorical if c in X.columns]
    col_numerical   = [c for c in col_numerical   if c in X.columns]

    # --- numerical → 32-bit binary string --------------------------------
    def numerical_to_binary(val: float, min_val: float, max_val: float) -> str:
        size = 32
        if max_val == min_val:
            return "0" * size
        normalized = (val - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))
        return format(int(normalized * (2 ** size - 1)), f"0{size}b")

    for col in col_numerical:
        if col not in metadata:
            metadata[col] = {"min": float(X[col].min()), "max": float(X[col].max())}
        min_val, max_val = metadata[col]["min"], metadata[col]["max"]
        X[col] = (
            X[col]
            .astype(object)
            .apply(lambda x: numerical_to_binary(x, min_val, max_val))
        )

    # --- categorical → fixed-width binary string -------------------------
    for col in col_categorical:
        if col not in metadata:
            metadata[col] = {
                "category_map": {cat: idx for idx, cat in enumerate(X[col].unique())}
            }
        category_map  = metadata[col]["category_map"]
        unique_values = len(category_map)
        size = int(np.ceil(np.log2(unique_values))) if unique_values > 1 else 1
        X[col] = (
            X[col]
            .astype(object)
            .apply(lambda x: format(category_map.get(x, 0), f"0{size}b"))
        )

    # --- join all binary strings per row → 1-D int8 array ---------------
    X_str = X.astype(str)
    rows_binary = []
    for i in range(len(X_str)):
        row_str = "".join(X_str.iloc[i].values)
        if any(c not in ("0", "1") for c in row_str):
            raise ValueError(
                f"Row {i} contains non-binary characters "
                f"{set(row_str) - {'0', '1'}}. "
                "A NaN was not filled — check your config."
            )
        rows_binary.append(
            np.frombuffer(row_str.encode(), dtype=np.uint8) - ord("0")
        )

    lengths = {len(r) for r in rows_binary}
    if len(lengths) > 1:
        raise ValueError(
            f"Rows have inconsistent binary lengths {lengths}. "
            "Check for unseen categories or NaN leakage."
        )

    return np.stack(rows_binary).astype(np.int8), y, metadata


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_models(task: str, hp: dict, seed: int = 0) -> dict:
    """Return {'LR': ..., 'DT': ..., 'RF': ...} for the given task."""
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


# ---------------------------------------------------------------------------
# Core evaluation  ← importable from a notebook
# ---------------------------------------------------------------------------

def evaluate(cfg: dict, verbose: bool = True) -> dict:
    """
    Run the full TSTR evaluation described in the config and return results.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config (same structure as the .yml files).
    verbose : bool
        Print a results table when True (default). Set to False for silent use
        in notebooks where you only need the returned dict.

    Returns
    -------
    dict with keys:
        dataset  : str
        task     : "classification" | "regression"
        metric   : "accuracy" | "mse"
        models   : {
            "LR": {"mean": float, "std": float, "scores": list[float]},
            "DT": {...},
            "RF": {...},
        }
    """
    dataset = cfg["dataset_name"].lower()
    task    = cfg["data"]["task"]
    target  = cfg["data"]["target_column"]

    assert dataset in HYPERPARAMS, (
        f"Unknown dataset '{dataset}'. Choose from: {list(HYPERPARAMS)}"
    )

    metric_fn    = accuracy_score if task == "classification" else mean_squared_error
    metric_label = "accuracy" if task == "classification" else "mse"
    raw_scores   = {name: [] for name in ["LR", "DT", "RF"]}
    syn_paths    = cfg["path_synthetic_trains"]

    for i, syn_path in enumerate(syn_paths, 1):
        if verbose:
            print(f"  [run {i}/{len(syn_paths)}] {syn_path}")

        # ── TSTR order ────────────────────────────────────────────────
        # 1. Fit encoding metadata on synthetic training data.
        # 2. Encode test set with that same metadata → identical bit widths.
        # ──────────────────────────────────────────────────────────────
        X_syn, y_syn, syn_meta = preprocess(load(syn_path), cfg, target, metadata={})
        X_test, y_test, _      = preprocess(load(cfg["path_test"]), cfg, target, metadata=syn_meta)

        if task == "classification":
            le = LabelEncoder().fit(np.concatenate([y_syn.values, y_test.values]))
            y_syn_enc  = le.transform(y_syn.values)
            y_test_enc = le.transform(y_test.values)
            scaler = None
        else:
            scaler     = MinMaxScaler()
            y_syn_enc  = scaler.fit_transform(y_syn.values.reshape(-1, 1)).flatten()
            y_test_enc = y_test.values   # kept in original scale; preds are inverse-transformed

        for name, model in build_models(task, HYPERPARAMS[dataset]).items():
            model.fit(X_syn, y_syn_enc)
            y_pred = model.predict(X_test)

            if task == "regression":
                y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

            raw_scores[name].append(float(metric_fn(y_test_enc, y_pred)))

    # Build structured result dict
    results = {
        "dataset": dataset,
        "task":    task,
        "metric":  metric_label,
        "models": {
            name: {
                "mean":   float(np.mean(s)),
                "std":    float(np.std(s)),
                "scores": s,
            }
            for name, s in raw_scores.items()
        },
    }

    if verbose:
        _print_results(results)

    return results


# ---------------------------------------------------------------------------
# Pretty printer (used by both evaluate() and the CLI)
# ---------------------------------------------------------------------------

def _print_results(results: dict) -> None:
    dataset      = results["dataset"].upper()
    task         = results["task"]
    metric_label = results["metric"].upper()

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} | Task: {task} | Metric: {metric_label}")
    print(f"{'='*60}")
    print(f"{'Model':<8}  {'Mean':>10}  {'Std':>10}  All scores")
    print(f"{'-'*60}")
    for name, info in results["models"].items():
        scores_str = "  ".join(f"{x:.4f}" for x in info["scores"])
        print(f"{name:<8}  {info['mean']:>10.4f}  {info['std']:>10.4f}  [{scores_str}]")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="TSTR evaluation for Binary Diffusion synthetic tabular data."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--quiet",  action="store_true", help="Suppress results table output.")
    return parser.parse_args()


def main():
    args    = _parse_args()
    cfg     = yaml.safe_load(open(args.config))
    results = evaluate(cfg, verbose=not args.quiet)
    return results


if __name__ == "__main__":
    main()
