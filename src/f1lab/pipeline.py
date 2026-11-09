"""End-to-end pipeline.

    python -m f1lab.synthetic        # or: python -m f1lab.ingest --years 2023 2024 2025
    python -m f1lab.pipeline
"""
from __future__ import annotations

import pandas as pd

from . import position_model as pm
from .config import MODEL_DIR, PROC_DIR, RAW_DIR, ensure_dirs
from .features import FEATURES, build_features
from .pace_model import build_pace_tables, circuit_profile
from .simulator import Driver, RaceParams, Strategy


def load_raw():
    return (pd.read_parquet(RAW_DIR / "results.parquet"),
            pd.read_parquet(RAW_DIR / "quali.parquet"),
            pd.read_parquet(RAW_DIR / "laps.parquet"))


def run() -> dict:
    ensure_dirs()
    results, quali, laps = load_raw()

    pace, comps = build_pace_tables(laps)
    pace.to_parquet(PROC_DIR / "pace.parquet", index=False)
    comps.to_parquet(PROC_DIR / "compounds.parquet", index=False)

    feats = build_features(results, quali, pace)
    feats.to_parquet(PROC_DIR / "features.parquet", index=False)

    oof = pm.walk_forward(feats)
    oof.to_parquet(PROC_DIR / "oof_predictions.parquet", index=False)
    metrics = pm.evaluate(oof)
    pm.save_metrics(metrics, PROC_DIR / "metrics.json")

    reg, _ = pm.fit_final(feats)
    imp = pm.shap_importance(reg, feats[FEATURES])
    imp.to_csv(PROC_DIR / "feature_importance.csv", index=False)
    return metrics


def scenario_from_data(circuit: str, season: int | None = None, pit_loss: float = 21.0,
                       sc_prob: float = 0.45, last_n_races: int = 5):
    """Build RaceParams + a 20-car field from the processed data.

    Driver pace = mean pace_delta over their last `last_n_races`; the grid is
    ordered by that pace (a stand-in for qualifying, replace with a real grid).
    """
    pace = pd.read_parquet(PROC_DIR / "pace.parquet")
    comps = pd.read_parquet(PROC_DIR / "compounds.parquet")
    prof = circuit_profile(comps, circuit).set_index("compound")
    params = RaceParams(
        n_laps=int(round(prof["n_race_laps"].mean())),
        base_lap=float(prof["base_lap"].mean()),
        pit_loss=pit_loss,
        offset=prof["offset"].fillna(0).to_dict(),
        deg=prof["deg_per_lap"].fillna(prof["deg_per_lap"].mean()).to_dict(),
        fuel_coef=float(prof["fuel_coef"].mean()),
        sc_prob=sc_prob,
    )
    pace = pace.sort_values(["season", "round"])
    if season:
        pace = pace[pace["season"] <= season]
    recent = pace.groupby("driver_code").tail(last_n_races).groupby("driver_code")["pace_delta"].mean().dropna()
    recent = recent.sort_values()
    default = Strategy(["MEDIUM", "HARD"], [params.n_laps // 2])
    field = [Driver(code, float(v), i + 1, default) for i, (code, v) in enumerate(recent.items())]
    return params, field


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
