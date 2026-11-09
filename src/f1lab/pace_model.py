"""Fuel- and tyre-corrected race pace.

For each race we fit one linear model on clean dry laps:

    lap_time = driver_effect
             + compound_offset[MEDIUM, HARD]        (vs SOFT)
             + deg[compound] * tyre_life            (s per lap of tyre age)
             + fuel_coef * laps_remaining           (s per lap of fuel still onboard)

`driver_effect` (relative to the field median) is the driver's *underlying race
pace*, with tyres, fuel and compound choice stripped out. The per-race
compound/degradation estimates feed the strategy simulator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLICKS = ["SOFT", "MEDIUM", "HARD"]
MIN_LAPS_DRIVER = 8
MIN_LAPS_COMPOUND = 15


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.copy()
    df = df[df["Compound"].isin(SLICKS)]
    df = df[df["lap_time_s"].notna()]
    df = df[(df["LapNumber"] > 1) & (~df["pit_in"].astype(bool)) & (~df["pit_out"].astype(bool))]
    df = df[df["TrackStatus"].astype(str) == "1"]
    if "rainfall" in df:
        df = df[~df["rainfall"].fillna(False).astype(bool)]
    # drop outliers within each driver-stint: > 107% of the stint median
    med = df.groupby(["season", "round", "Driver", "Stint"])["lap_time_s"].transform("median")
    return df[df["lap_time_s"] <= 1.07 * med]


def fit_race(df: pd.DataFrame) -> dict | None:
    """df: clean laps of ONE race. Returns driver pace + compound params."""
    drivers = sorted(df["Driver"].unique())
    n_laps = int(df["LapNumber"].max())
    cols, names = [], []

    for d in drivers:
        cols.append((df["Driver"] == d).astype(float).to_numpy())
        names.append(("drv", d))
    for c in ("MEDIUM", "HARD"):
        cols.append((df["Compound"] == c).astype(float).to_numpy())
        names.append(("off", c))
    for c in SLICKS:
        mask = (df["Compound"] == c).to_numpy()
        if mask.sum() >= MIN_LAPS_COMPOUND:
            cols.append(np.where(mask, df["TyreLife"].to_numpy(), 0.0))
            names.append(("deg", c))
    cols.append((n_laps - df["LapNumber"]).to_numpy(dtype=float))
    names.append(("fuel", "all"))

    X = np.column_stack(cols)
    y = df["lap_time_s"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef = dict(zip(names, beta))

    counts = df.groupby("Driver").size()
    drv_eff = pd.Series({d: coef[("drv", d)] for d in drivers})
    drv_eff[counts.reindex(drv_eff.index).fillna(0) < MIN_LAPS_DRIVER] = np.nan
    pace = (drv_eff - drv_eff.median()).rename("pace_delta")

    comp_rows = []
    for c in SLICKS:
        comp_rows.append(
            dict(
                compound=c,
                offset=0.0 if c == "SOFT" else coef.get(("off", c), np.nan),
                deg_per_lap=coef.get(("deg", c), np.nan),
                n_laps=int((df["Compound"] == c).sum()),
            )
        )
    return dict(
        pace=pace,
        compounds=pd.DataFrame(comp_rows),
        fuel_coef=float(coef[("fuel", "all")]),
        base_lap=float(drv_eff.median()),
        n_laps=n_laps,
    )


def build_pace_tables(laps: pd.DataFrame):
    """Returns (driver pace per race, compound params per race)."""
    clean = clean_laps(laps)
    pace_rows, comp_rows = [], []
    for (season, rnd), g in clean.groupby(["season", "round"]):
        fit = fit_race(g)
        if fit is None:
            continue
        p = fit["pace"].reset_index()
        p.columns = ["driver_code", "pace_delta"]
        p["season"], p["round"] = season, rnd
        p["n_clean_laps"] = p["driver_code"].map(g.groupby("Driver").size())
        pace_rows.append(p)
        c = fit["compounds"].copy()
        c["season"], c["round"] = season, rnd
        c["fuel_coef"], c["base_lap"], c["n_race_laps"] = fit["fuel_coef"], fit["base_lap"], fit["n_laps"]
        c["circuit"] = g["circuit"].iloc[0]
        comp_rows.append(c)
    pace = pd.concat(pace_rows, ignore_index=True) if pace_rows else pd.DataFrame()
    comps = pd.concat(comp_rows, ignore_index=True) if comp_rows else pd.DataFrame()
    return pace, comps


def circuit_profile(comps: pd.DataFrame, circuit: str) -> pd.DataFrame:
    """Average compound parameters for a circuit over all seasons (fallback: all circuits)."""
    sub = comps[comps["circuit"] == circuit]
    if sub.empty:
        sub = comps
    return sub.groupby("compound", as_index=False)[["offset", "deg_per_lap", "fuel_coef", "base_lap", "n_race_laps"]].mean()
