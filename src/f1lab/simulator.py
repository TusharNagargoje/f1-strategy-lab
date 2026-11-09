"""Vectorised Monte Carlo race simulator for pit-strategy comparison.

Each simulation draws one shared safety-car window (it affects the whole field),
per-lap noise per driver, and DNFs. A driver's race time is the sum of lap times:

    base + pace_delta + compound_offset + deg * tyre_age + fuel * laps_remaining + noise

Pitting under the safety car costs only a fraction of the normal pit loss, which
is what makes SC probability matter for strategy choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class RaceParams:
    n_laps: int
    base_lap: float
    pit_loss: float
    offset: dict           # compound -> fresh-tyre pace offset (s)
    deg: dict              # compound -> degradation (s per lap of tyre age)
    fuel_coef: float = 0.035
    sc_prob: float = 0.45   # P(at least one safety car in the race)
    sc_len: int = 4         # laps under SC
    sc_lap_factor: float = 1.35
    sc_pit_discount: float = 0.45   # fraction of pit loss paid when pitting under SC
    noise_sd: float = 0.25
    start_slot_loss: float = 0.12   # lap-1 time lost per grid slot
    dnf_prob: float = 0.05


@dataclass
class Strategy:
    compounds: list
    pit_laps: list   # lap numbers on which the driver pits (end of lap)

    def label(self) -> str:
        parts = [self.compounds[0][0]]
        for lap, c in zip(self.pit_laps, self.compounds[1:]):
            parts.append(f"-{lap}-{c[0]}")
        return "".join(parts)


@dataclass
class Driver:
    name: str
    pace_delta: float
    grid: int
    strategy: Strategy


def _stint_arrays(p: RaceParams, s: Strategy):
    """Deterministic per-lap compound index, tyre age, and pit flag for a strategy."""
    comp_idx = np.zeros(p.n_laps, dtype=int)
    age = np.zeros(p.n_laps)
    pit = np.zeros(p.n_laps, dtype=bool)
    stint, a = 0, 0
    pits = set(s.pit_laps)
    for lap in range(1, p.n_laps + 1):
        a += 1
        comp_idx[lap - 1] = stint
        age[lap - 1] = a
        if lap in pits:
            pit[lap - 1] = True
            stint += 1
            a = 0
    return comp_idx, age, pit


def draw_safety_cars(p: RaceParams, sims: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean (sims, n_laps) mask of laps run under the safety car."""
    mask = np.zeros((sims, p.n_laps), dtype=bool)
    has = rng.random(sims) < p.sc_prob
    starts = rng.integers(3, max(4, p.n_laps - p.sc_len - 2), size=sims)
    for i in np.flatnonzero(has):
        mask[i, starts[i]: starts[i] + p.sc_len] = True
    return mask


def _clean_laps(p: RaceParams, d: Driver) -> np.ndarray:
    """Noise-free, safety-car-free lap times for a driver's strategy (pit loss excluded)."""
    comp_idx, age, _ = _stint_arrays(p, d.strategy)
    comps = d.strategy.compounds
    off = np.array([p.offset[c] for c in comps])[comp_idx]
    deg = np.array([p.deg[c] for c in comps])[comp_idx]
    laps = np.arange(1, p.n_laps + 1)
    return p.base_lap + d.pace_delta + off + deg * age + p.fuel_coef * (p.n_laps - laps)


def lap_times_matrix(p: RaceParams, d: Driver, sc_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulated lap times, shape (sims, n_laps), including start loss, SC laps and pit stops."""
    sims = sc_mask.shape[0]
    _, _, pit = _stint_arrays(p, d.strategy)
    clean = _clean_laps(p, d)
    lap_t = clean[None, :] + rng.normal(0, p.noise_sd, size=(sims, p.n_laps))
    lap_t = np.where(sc_mask, p.base_lap * p.sc_lap_factor + rng.normal(0, 0.3, size=lap_t.shape), lap_t)
    lap_t[:, 0] += 5.0 + p.start_slot_loss * (d.grid - 1)
    pit_cost = np.where(sc_mask, p.pit_loss * p.sc_pit_discount, p.pit_loss)
    return lap_t + pit_cost * pit[None, :]


def simulate_times(p: RaceParams, d: Driver, sc_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Total race time per simulation for one driver (inf if DNF)."""
    sims = sc_mask.shape[0]
    total = lap_times_matrix(p, d, sc_mask, rng).sum(axis=1)
    dnf = rng.random(sims) < p.dnf_prob
    return np.where(dnf, np.inf, total)


def expected_lap_trace(p: RaceParams, d: Driver) -> np.ndarray:
    """Expected lap times with no noise or safety car; pit laps include the full pit loss."""
    _, _, pit = _stint_arrays(p, d.strategy)
    t = _clean_laps(p, d) + p.pit_loss * pit
    t[0] += 5.0 + p.start_slot_loss * (d.grid - 1)
    return t


def random_strategy(p: RaceParams, rng: np.random.Generator) -> Strategy:
    """A plausible dry-race strategy (mostly one-stop, sometimes two) for filling out a field."""
    pairs = [("MEDIUM", "HARD"), ("SOFT", "HARD"), ("SOFT", "MEDIUM"), ("HARD", "MEDIUM")]
    if rng.random() < 0.25:
        seqs = [("SOFT", "MEDIUM", "HARD"), ("MEDIUM", "HARD", "MEDIUM"), ("SOFT", "HARD", "MEDIUM")]
        seq = seqs[int(rng.integers(len(seqs)))]
        a = int(p.n_laps * rng.uniform(0.2, 0.32))
        b = int(p.n_laps * rng.uniform(0.55, 0.7))
        return Strategy(list(seq), [a, b])
    seq = pairs[int(rng.integers(len(pairs)))]
    return Strategy(list(seq), [int(p.n_laps * rng.uniform(0.33, 0.6))])


def stint_segments(p: RaceParams, s: Strategy) -> list:
    """[(compound, first_lap, last_lap)] for drawing a stint timeline."""
    bounds = [0] + list(s.pit_laps) + [p.n_laps]
    return [(c, bounds[i] + 1, bounds[i + 1]) for i, c in enumerate(s.compounds)]


def race_trace(p: RaceParams, drivers: list, seed: int = 0) -> dict:
    """One simulated race, lap by lap, for a replay view.

    Returns cumulative times (n_drivers, n_laps) with np.inf after a retirement,
    positions (n_drivers, n_laps) with NaN once retired, and the SC / DNF info.
    """
    rng = np.random.default_rng(seed)
    sc = draw_safety_cars(p, 1, rng)
    laps = np.stack([lap_times_matrix(p, d, sc, rng)[0] for d in drivers])
    cum = laps.cumsum(axis=1)
    n = len(drivers)
    dnf_lap = np.full(n, -1)
    for i in np.flatnonzero(rng.random(n) < p.dnf_prob):
        dnf_lap[i] = int(rng.integers(3, p.n_laps))
        cum[i, dnf_lap[i]:] = np.inf
    pos = np.full(cum.shape, np.nan)
    for lap in range(p.n_laps):
        alive = np.flatnonzero(np.isfinite(cum[:, lap]))
        order = alive[np.argsort(cum[alive, lap], kind="stable")]
        pos[order, lap] = np.arange(1, len(order) + 1)
    return dict(cum=cum, pos=pos, sc=sc[0], dnf_lap=dnf_lap)


def simulate_race(p: RaceParams, drivers: list, sims: int = 2000, seed: int = 0):
    """Returns finishing-position matrix (sims, n_drivers) and the times matrix."""
    rng = np.random.default_rng(seed)
    sc = draw_safety_cars(p, sims, rng)
    times = np.column_stack([simulate_times(p, d, sc, rng) for d in drivers])
    order = np.argsort(times, axis=1, kind="stable")
    pos = np.empty_like(order)
    rows = np.arange(sims)[:, None]
    pos[rows, order] = np.arange(1, len(drivers) + 1)[None, :]
    return pos, times


def summarize(pos: np.ndarray, drivers: list) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(drivers):
        col = pos[:, i]
        rows.append(dict(
            driver=d.name, grid=d.grid, strategy=d.strategy.label(),
            mean_pos=col.mean(), win_p=(col == 1).mean(), podium_p=(col <= 3).mean(),
            points_p=(col <= 10).mean(),
        ))
    return pd.DataFrame(rows).sort_values("mean_pos").reset_index(drop=True)


def enumerate_strategies(p: RaceParams, compounds=("SOFT", "MEDIUM", "HARD"), max_stops: int = 2,
                         window_step: int = 4, min_stint: int = 8) -> list:
    """All dry-race-legal strategies (>=2 distinct compounds) on a coarse pit-lap grid."""
    out = []
    candidates = list(range(min_stint, p.n_laps - min_stint + 1, window_step))
    for stops in range(1, max_stops + 1):
        for laps in combinations(candidates, stops):
            if any(b - a < min_stint for a, b in zip(laps, laps[1:])):
                continue
            seqs = _compound_sequences(compounds, stops + 1)
            for seq in seqs:
                out.append(Strategy(list(seq), list(laps)))
    return out


def _compound_sequences(compounds, n):
    seqs = [[]]
    for _ in range(n):
        seqs = [s + [c] for s in seqs for c in compounds if not s or s[-1] != c]
    return [s for s in seqs if len(set(s)) >= 2]


def compare_strategies(p: RaceParams, field_drivers: list, target: str, candidates: list,
                       sims: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Evaluate each candidate strategy for `target` against a fixed field (common random numbers)."""
    idx = next(i for i, d in enumerate(field_drivers) if d.name == target)
    base = field_drivers[idx]
    rows = []
    for s in candidates:
        drivers = list(field_drivers)
        drivers[idx] = Driver(base.name, base.pace_delta, base.grid, s)
        pos, times = simulate_race(p, drivers, sims=sims, seed=seed)
        col = pos[:, idx]
        t = times[:, idx]
        t = t[np.isfinite(t)]
        rows.append(dict(
            strategy=s.label(), stops=len(s.pit_laps), mean_pos=col.mean(),
            podium_p=(col <= 3).mean(), points_p=(col <= 10).mean(),
            mean_time=t.mean() if len(t) else np.nan, time_sd=t.std() if len(t) else np.nan,
        ))
    return pd.DataFrame(rows).sort_values(["mean_pos", "mean_time"]).reset_index(drop=True)
