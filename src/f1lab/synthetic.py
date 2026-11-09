"""Synthetic F1-like data so the whole pipeline runs offline.

Schema matches the real ingestion output (results / quali / laps parquet files).
Numbers are fabricated but follow realistic structure: car strength dominates,
driver skill adds a smaller term, tyres degrade linearly, fuel burns off,
and a safety car appears at random. Use it to develop and test; switch to real
data with `python -m f1lab.ingest`.

    python -m f1lab.synthetic
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_DIR, RANDOM_SEED, ensure_dirs

TEAMS = {  # car pace advantage in seconds per lap (lower = faster)
    "alpha": 0.0, "bravo": 0.25, "charlie": 0.45, "delta": 0.7, "echo": 0.95,
    "foxtrot": 1.2, "golf": 1.5, "hotel": 1.8, "india": 2.1, "juliet": 2.5,
}
CIRCUITS = {  # base lap (s), laps, tyre deg (s/lap) for SOFT/MEDIUM/HARD, pit loss (s)
    "monza_like": (82.0, 53, (0.11, 0.07, 0.04), 21.0),
    "monaco_like": (74.0, 60, (0.08, 0.05, 0.03), 19.0),
    "spa_like": (106.0, 44, (0.13, 0.08, 0.05), 20.0),
    "suzuka_like": (91.0, 53, (0.14, 0.09, 0.05), 22.0),
    "bahrain_like": (96.0, 57, (0.16, 0.10, 0.06), 23.0),
}
COMPOUND_OFFSET = {"SOFT": 0.0, "MEDIUM": 0.6, "HARD": 1.1}  # fresh-tyre pace delta
FUEL_EFFECT = 0.035  # s/lap lost per lap of fuel still on board (~0.035 s/kg-ish)


def _drivers(rng: np.random.Generator):
    drivers = []
    for ti, team in enumerate(TEAMS):
        for j in range(2):
            drivers.append((f"{team}_{j+1}", team, rng.normal(0, 0.18)))
    return drivers


def generate(seasons=(2023, 2024, 2025), rounds_per_season=10, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    drivers = _drivers(rng)
    circuit_names = list(CIRCUITS)
    results, quali, laps_out = [], [], []

    for season in seasons:
        # car pace drifts each season
        drift = {t: rng.normal(0, 0.15) for t in TEAMS}
        for rnd in range(1, rounds_per_season + 1):
            circuit = circuit_names[(rnd - 1) % len(circuit_names)]
            base, n_laps, deg, pit_loss = CIRCUITS[circuit]
            rain = rng.random() < 0.08
            sc_lap = int(rng.integers(8, n_laps - 8)) if rng.random() < 0.45 else None

            # qualifying
            qpace = {
                d: TEAMS[t] + drift[t] + skill + rng.normal(0, 0.12)
                for d, t, skill in drivers
            }
            q_order = sorted(qpace, key=qpace.get)
            for pos, d in enumerate(q_order, 1):
                quali.append(dict(season=season, round=rnd, driver_id=d, quali_pos=pos))
            grid = {d: i + 1 for i, d in enumerate(q_order)}

            # race simulation, lap by lap
            race_times = {}
            for d, team, skill in drivers:
                pace = TEAMS[team] + drift[team] + skill
                dnf_lap = int(rng.integers(5, n_laps)) if rng.random() < 0.06 else None
                n_stops = int(rng.choice([1, 2], p=[0.7, 0.3]))
                pit_laps = sorted(
                    rng.integers(12, n_laps - 10, size=n_stops).tolist()
                )
                comps = list(rng.choice(COMPOUNDS_ALL, size=n_stops + 1))
                if len(set(comps)) == 1:
                    comps[-1] = "HARD" if comps[0] != "HARD" else "MEDIUM"
                total, stint, tyre_age = 0.0, 1, 0
                comp = comps[0]
                lap_rows = []
                for lap in range(1, n_laps + 1):
                    if dnf_lap and lap >= dnf_lap:
                        break
                    tyre_age += 1
                    ci = ["SOFT", "MEDIUM", "HARD"].index(comp)
                    t = (
                        base + pace + COMPOUND_OFFSET[comp]
                        + deg[ci] * tyre_age
                        + FUEL_EFFECT * (n_laps - lap)
                        + rng.normal(0, 0.15)
                        + (6.0 if rain else 0.0)
                    )
                    status = "1"
                    if sc_lap and sc_lap <= lap < sc_lap + 4:
                        t = base * 1.35 + rng.normal(0, 0.3)
                        status = "4"
                    pit_in = lap in pit_laps
                    if pit_in:
                        t += pit_loss
                    if lap == 1:
                        t += 6.0 + 0.15 * (grid[d] - 1) * 0.3
                    total += t
                    lap_rows.append(
                        dict(
                            season=season, round=rnd, circuit=circuit, Driver=d, Team=team,
                            LapNumber=lap, lap_time_s=t, Stint=stint, Compound=comp,
                            TyreLife=tyre_age, TrackStatus=status, pit_in=pit_in,
                            pit_out=False, IsAccurate=(lap > 1 and status == "1" and not pit_in),
                            rainfall=rain, track_temp=float(rng.normal(35, 4)),
                            air_temp=float(rng.normal(25, 3)),
                        )
                    )
                    if pit_in:
                        stint += 1
                        tyre_age = 0
                        comp = comps[min(stint - 1, len(comps) - 1)]
                laps_out.extend(lap_rows)
                finished = dnf_lap is None
                race_times[d] = (total if finished else 1e9 + (n_laps - len(lap_rows)), finished,
                                 len(lap_rows))

            order = sorted(race_times, key=lambda k: race_times[k][0])
            pts = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 10
            team_of = {d: t for d, t, _ in drivers}
            for pos, d in enumerate(order, 1):
                results.append(
                    dict(
                        season=season, round=rnd, circuit_id=circuit, race_name=f"{circuit} GP",
                        date=f"{season}-{rnd:02d}-15", driver_id=d, driver_code=d,
                        constructor_id=team_of[d],
                        grid=grid[d], position=pos,
                        status="Finished" if race_times[d][1] else "Retired",
                        points=float(pts[pos - 1]), laps=race_times[d][2],
                    )
                )
    return pd.DataFrame(results), pd.DataFrame(quali), pd.DataFrame(laps_out)


COMPOUNDS_ALL = ["SOFT", "MEDIUM", "HARD"]


def main():
    ensure_dirs()
    results, quali, laps = generate()
    results.to_parquet(RAW_DIR / "results.parquet", index=False)
    quali.to_parquet(RAW_DIR / "quali.parquet", index=False)
    laps.to_parquet(RAW_DIR / "laps.parquet", index=False)
    print(f"synthetic: results={len(results)} quali={len(quali)} laps={len(laps)}")


if __name__ == "__main__":
    main()
