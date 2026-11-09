"""Data ingestion.

Real data:
    python -m f1lab.ingest --years 2022 2023 2024 2025 2026

Pulls race results + qualifying from the Jolpica-F1 API and lap-level data
(tyre compound, tyre life, lap time, track status) from FastF1. Outputs Parquet
files in data/raw/.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import requests

from .config import CACHE_DIR, JOLPICA_BASE, RAW_DIR, ensure_dirs


# ---------------------------------------------------------------- Jolpica ---
def _get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()
    return {}


def _paged(endpoint: str, table: str, key: str, limit: int = 100) -> list[dict]:
    out, offset = [], 0
    while True:
        js = _get(f"{JOLPICA_BASE}/{endpoint}.json", {"limit": limit, "offset": offset})
        mr = js["MRData"]
        out.extend(mr[table][key])
        offset += limit
        if offset >= int(mr["total"]):
            return out


def fetch_results(year: int) -> pd.DataFrame:
    races = _paged(f"{year}/results", "RaceTable", "Races")
    rows = []
    for race in races:
        for res in race["Results"]:
            rows.append(
                dict(
                    season=int(race["season"]),
                    round=int(race["round"]),
                    circuit_id=race["Circuit"]["circuitId"],
                    race_name=race["raceName"],
                    date=race["date"],
                    driver_id=res["Driver"]["driverId"],
                    driver_code=res["Driver"].get("code"),
                    constructor_id=res["Constructor"]["constructorId"],
                    grid=int(res["grid"]),
                    position=int(res["position"]),
                    status=res["status"],
                    points=float(res["points"]),
                    laps=int(res["laps"]),
                )
            )
    return pd.DataFrame(rows)


def fetch_qualifying(year: int) -> pd.DataFrame:
    races = _paged(f"{year}/qualifying", "RaceTable", "Races")
    rows = []
    for race in races:
        for q in race["QualifyingResults"]:
            rows.append(
                dict(
                    season=int(race["season"]),
                    round=int(race["round"]),
                    driver_id=q["Driver"]["driverId"],
                    quali_pos=int(q["position"]),
                    q1=q.get("Q1"),
                    q2=q.get("Q2"),
                    q3=q.get("Q3"),
                )
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- FastF1 ---
def fetch_laps(year: int, rounds: list[int] | None = None) -> pd.DataFrame:
    import fastf1  # imported lazily so the rest of the package works without it

    ensure_dirs()
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    frames = []
    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        if rounds and rnd not in rounds:
            continue
        try:
            session = fastf1.get_session(year, rnd, "R")
            session.load(telemetry=False, weather=True, messages=False)
        except Exception as exc:  # session not run yet / data missing
            print(f"skip {year} R{rnd}: {exc}")
            continue
        laps = session.laps.copy()
        keep = [
            "Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Stint",
            "Compound", "TyreLife", "FreshTyre", "PitInTime", "PitOutTime",
            "TrackStatus", "IsAccurate", "Position",
        ]
        laps = laps[[c for c in keep if c in laps.columns]]
        laps["lap_time_s"] = laps["LapTime"].dt.total_seconds()
        laps["pit_in"] = laps["PitInTime"].notna()
        laps["pit_out"] = laps["PitOutTime"].notna()
        laps = laps.drop(columns=["LapTime", "PitInTime", "PitOutTime"])
        laps["season"], laps["round"] = year, rnd
        laps["circuit"] = ev["Location"]
        w = session.weather_data
        if w is not None and len(w):
            laps["air_temp"] = float(w["AirTemp"].mean())
            laps["track_temp"] = float(w["TrackTemp"].mean())
            laps["rainfall"] = bool(w["Rainfall"].any())
        frames.append(laps)
        print(f"loaded {year} R{rnd} {ev['EventName']} ({len(laps)} laps)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--skip-laps", action="store_true", help="results/quali only")
    args = ap.parse_args()

    ensure_dirs()
    results = pd.concat([fetch_results(y) for y in args.years], ignore_index=True)
    quali = pd.concat([fetch_qualifying(y) for y in args.years], ignore_index=True)
    results.to_parquet(RAW_DIR / "results.parquet", index=False)
    quali.to_parquet(RAW_DIR / "quali.parquet", index=False)
    print(f"results: {len(results)} rows, quali: {len(quali)} rows")

    if not args.skip_laps:
        laps = pd.concat([fetch_laps(y) for y in args.years], ignore_index=True)
        laps.to_parquet(RAW_DIR / "laps.parquet", index=False)
        print(f"laps: {len(laps)} rows")


if __name__ == "__main__":
    main()
