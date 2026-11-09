"""Finishing-position and podium models with walk-forward (time-ordered) validation."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .config import MODEL_DIR, RANDOM_SEED
from .features import FEATURES


def _reg():
    from lightgbm import LGBMRegressor
    # deliberately small/regularised: ~20 rows per race means big trees just fit noise
    return LGBMRegressor(
        n_estimators=80, learning_rate=0.03, num_leaves=4, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
        random_state=RANDOM_SEED, verbose=-1,
    )


def _clf():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=10,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=RANDOM_SEED, verbose=-1,
    )


def walk_forward(df: pd.DataFrame, min_train_races: int = 12, step: int = 4) -> pd.DataFrame:
    """Train on races [0, k), predict races [k, k+step). Returns out-of-fold predictions."""
    n_races = int(df["race_idx"].max()) + 1
    parts = []
    for k in range(min_train_races, n_races, step):
        tr = df[df["race_idx"] < k]
        te = df[(df["race_idx"] >= k) & (df["race_idx"] < k + step)]
        if te.empty:
            break
        reg, clf = _reg(), _clf()
        # model the shift from the grid slot, so the grid-order baseline is the starting point
        reg.fit(tr[FEATURES], tr["position"] - tr["grid"])
        clf.fit(tr[FEATURES], tr["podium"])
        out = te[["season", "round", "race_idx", "driver_id", "grid", "position", "podium"]].copy()
        out["pred_position"] = te["grid"].to_numpy() + reg.predict(te[FEATURES])
        out["pred_podium_p"] = clf.predict_proba(te[FEATURES])[:, 1]
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def evaluate(oof: pd.DataFrame) -> dict:
    """Compare model with the grid-order baseline."""
    def per_race(frame, col):
        rho, top3, win = [], [], []
        for _, r in frame.groupby("race_idx"):
            if r[col].nunique() > 1:
                rho.append(spearmanr(r[col], r["position"]).statistic)
            pred_order = r.sort_values(col)
            actual_podium = set(r.loc[r["position"] <= 3, "driver_id"])
            top3.append(len(actual_podium & set(pred_order["driver_id"].head(3))) / max(len(actual_podium), 1))
            win.append(float(pred_order["driver_id"].iloc[0] == r.sort_values("position")["driver_id"].iloc[0]))
        return dict(
            mae=float(np.mean(np.abs(frame[col] - frame["position"]))),
            spearman=float(np.nanmean(rho)),
            podium_overlap=float(np.mean(top3)),
            winner_acc=float(np.mean(win)),
        )
    metrics = {
        "n_test_races": int(oof["race_idx"].nunique()),
        "model": per_race(oof, "pred_position"),
        "grid_baseline": per_race(oof.assign(base=oof["grid"]), "base"),
    }
    if oof["podium"].nunique() > 1:
        metrics["model"]["podium_auc"] = float(roc_auc_score(oof["podium"], oof["pred_podium_p"]))
        metrics["grid_baseline"]["podium_auc"] = float(roc_auc_score(oof["podium"], -oof["grid"]))
    return metrics


def fit_final(df: pd.DataFrame):
    import joblib
    reg, clf = _reg(), _clf()
    reg.fit(df[FEATURES], df["position"] - df["grid"])  # predicts shift from grid; add grid back
    clf.fit(df[FEATURES], df["podium"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, MODEL_DIR / "position_reg.joblib")
    joblib.dump(clf, MODEL_DIR / "podium_clf.joblib")
    return reg, clf


def shap_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    try:
        import shap
        vals = shap.TreeExplainer(model).shap_values(X)
        imp = np.abs(vals).mean(axis=0)
    except Exception:  # shap missing or incompatible -> fall back to gain
        imp = model.booster_.feature_importance(importance_type="gain")
        imp = imp / imp.sum()
    return pd.DataFrame({"feature": X.columns, "importance": imp}).sort_values("importance", ascending=False)


def save_metrics(metrics: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
