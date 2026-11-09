import pandas as pd

from f1lab.features import FEATURES, build_features
from f1lab.pace_model import build_pace_tables
from f1lab.synthetic import generate


def _data():
    return generate(seasons=(2023, 2024), rounds_per_season=5, seed=1)


def test_features_have_no_future_leakage():
    results, quali, laps = _data()
    pace, _ = build_pace_tables(laps)
    df = build_features(results, quali, pace)
    first = df[df["race_idx"] == 0]
    # no history before the first race
    assert first["drv_form_pos"].isna().all()
    assert first["team_form_pos"].isna().all()
    # changing the final race's results must not change any of its own features
    last_idx = df["race_idx"].max()
    r2 = results.copy()
    mask = (r2["season"] == df.loc[df.race_idx == last_idx, "season"].iloc[0]) & \
           (r2["round"] == df.loc[df.race_idx == last_idx, "round"].iloc[0])
    r2.loc[mask, "position"] = r2.loc[mask, "position"].max() + 1 - r2.loc[mask, "position"]
    df2 = build_features(r2, quali, pace)
    a = df[df.race_idx == last_idx].set_index("driver_id")[FEATURES].sort_index()
    b = df2[df2.race_idx == last_idx].set_index("driver_id")[FEATURES].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_pace_model_recovers_tyre_ordering():
    _, _, laps = _data()
    _, comps = build_pace_tables(laps)
    avg = comps.groupby("compound")["deg_per_lap"].mean()
    assert avg["SOFT"] > avg["HARD"]  # softs degrade faster in the synthetic world
