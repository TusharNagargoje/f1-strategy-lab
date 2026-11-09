"""Leakage-safe feature engineering for finishing-position prediction.

Every rolling feature is shifted by one race so a row only sees information that
was available *before* that race started. Qualifying position and grid are known
pre-race, so they are used directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "grid", "quali_pos", "quali_vs_teammate",
    "drv_form_pos", "drv_form_quali", "drv_form_gain", "drv_dnf_rate",
    "team_form_pos", "team_form_quali",
    "circuit_hist_pos",
    "drv_pace_form", "team_pace_form",
]


def _roll(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()


def build_features(results: pd.DataFrame, quali: pd.DataFrame, pace: pd.DataFrame | None = None) -> pd.DataFrame:
    df = results.merge(
        quali[["season", "round", "driver_id", "quali_pos"]],
        on=["season", "round", "driver_id"], how="left",
    )
    df["quali_pos"] = df["quali_pos"].fillna(df["grid"].replace(0, np.nan)).fillna(20)
    # pit-lane starts are reported as grid 0
    df["grid"] = df["grid"].replace(0, 20)

    df = df.sort_values(["season", "round", "position"]).reset_index(drop=True)
    df["race_idx"] = df.groupby(["season", "round"], sort=True).ngroup()
    s = df["status"].astype(str)
    df["dnf"] = (~(s.eq("Finished") | s.str.startswith("+"))).astype(int)
    df["gain"] = df["grid"] - df["position"]

    if pace is not None and len(pace):
        if "driver_code" not in df:
            df["driver_code"] = df["driver_id"]
        df = df.merge(pace[["season", "round", "driver_code", "pace_delta"]],
                      on=["season", "round", "driver_code"], how="left")
    else:
        df["pace_delta"] = np.nan

    g = df.groupby("driver_id", sort=False)
    df["drv_form_pos"] = g["position"].transform(lambda x: _roll(x, 5))
    df["drv_form_quali"] = g["quali_pos"].transform(lambda x: _roll(x, 5))
    df["drv_form_gain"] = g["gain"].transform(lambda x: _roll(x, 8))
    df["drv_dnf_rate"] = g["dnf"].transform(lambda x: _roll(x, 15))
    df["drv_pace_form"] = g["pace_delta"].transform(lambda x: _roll(x, 4))

    # teammate qualifying delta (known pre-race)
    team_mean_q = df.groupby(["race_idx", "constructor_id"])["quali_pos"].transform("mean")
    n_team = df.groupby(["race_idx", "constructor_id"])["quali_pos"].transform("count")
    df["quali_vs_teammate"] = np.where(n_team > 1, df["quali_pos"] - (team_mean_q * n_team - df["quali_pos"]) / (n_team - 1).clip(lower=1), 0.0)

    # constructor-level form (rolling over previous races)
    tr = (df.groupby(["race_idx", "constructor_id"], as_index=False)
            .agg(t_pos=("position", "mean"), t_q=("quali_pos", "mean"), t_pace=("pace_delta", "mean"))
            .sort_values("race_idx"))
    tg = tr.groupby("constructor_id", sort=False)
    tr["team_form_pos"] = tg["t_pos"].transform(lambda x: _roll(x, 5))
    tr["team_form_quali"] = tg["t_q"].transform(lambda x: _roll(x, 5))
    tr["team_pace_form"] = tg["t_pace"].transform(lambda x: _roll(x, 4))
    df = df.merge(tr[["race_idx", "constructor_id", "team_form_pos", "team_form_quali", "team_pace_form"]],
                  on=["race_idx", "constructor_id"], how="left")

    # driver's history at this circuit (previous visits only)
    df = df.sort_values(["race_idx", "position"]).reset_index(drop=True)
    df["circuit_hist_pos"] = (df.groupby(["driver_id", "circuit_id"])["position"]
                                .transform(lambda x: x.shift(1).expanding().mean()))

    df["podium"] = (df["position"] <= 3).astype(int)
    df["points_finish"] = (df["position"] <= 10).astype(int)
    return df
