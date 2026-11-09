import numpy as np

from f1lab.simulator import (Driver, RaceParams, Strategy, compare_strategies,
                             draw_safety_cars, enumerate_strategies, simulate_race)


def params(**kw):
    base = dict(n_laps=50, base_lap=90.0, pit_loss=21.0,
                offset={"SOFT": 0.0, "MEDIUM": 0.6, "HARD": 1.1},
                deg={"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}, dnf_prob=0.0, sc_prob=0.0)
    base.update(kw)
    return RaceParams(**base)


def field(p, n=10):
    s = Strategy(["MEDIUM", "HARD"], [25])
    return [Driver(f"D{i}", i * 0.3, i + 1, s) for i in range(n)]


def test_positions_are_a_permutation():
    p = params()
    pos, _ = simulate_race(p, field(p), sims=200, seed=1)
    for row in pos:
        assert sorted(row) == list(range(1, 11))


def test_faster_driver_finishes_ahead_on_average():
    p = params()
    pos, _ = simulate_race(p, field(p), sims=500, seed=2)
    means = pos.mean(axis=0)
    assert means[0] < means[-1]


def test_more_pit_loss_hurts_two_stoppers_more():
    p_cheap, p_costly = params(pit_loss=15.0), params(pit_loss=30.0)
    one, two = Strategy(["MEDIUM", "HARD"], [25]), Strategy(["SOFT", "MEDIUM", "HARD"], [14, 32])
    def gap(p):
        f = field(p, 6)
        a = compare_strategies(p, f, "D3", [one, two], sims=400, seed=3).set_index("strategy")
        return a.loc[two.label(), "mean_time"] - a.loc[one.label(), "mean_time"]
    assert gap(p_costly) > gap(p_cheap)


def test_safety_car_mask_shape_and_rate():
    p = params(sc_prob=0.5)
    m = draw_safety_cars(p, 4000, np.random.default_rng(0))
    assert m.shape == (4000, p.n_laps)
    rate = m.any(axis=1).mean()
    assert 0.44 < rate < 0.56


def test_enumerated_strategies_use_two_compounds():
    p = params()
    cands = enumerate_strategies(p)
    assert cands and all(len(set(s.compounds)) >= 2 for s in cands)


def test_race_trace_positions_are_valid_each_lap():
    from f1lab.simulator import race_trace
    p = params(dnf_prob=0.15, sc_prob=1.0)
    d = field(p, 10)
    tr = race_trace(p, d, seed=4)
    assert tr["pos"].shape == (10, p.n_laps) and tr["sc"].any()
    for lap in range(p.n_laps):
        col = tr["pos"][:, lap]
        alive = col[~np.isnan(col)]
        assert sorted(alive) == list(range(1, len(alive) + 1))


def test_stint_segments_cover_race_and_trace_has_pit_spike():
    from f1lab.simulator import expected_lap_trace, stint_segments
    p = params()
    d = field(p, 3)[1]
    segs = stint_segments(p, d.strategy)
    assert segs[0][1] == 1 and segs[-1][2] == p.n_laps
    assert all(a[2] + 1 == b[1] for a, b in zip(segs, segs[1:]))
    t = expected_lap_trace(p, d)
    assert t[24] > t[23] + 15  # lap 25 is the pit lap
