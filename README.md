# F1 Strategy Lab

Race-outcome prediction + Monte Carlo pit-strategy simulation, in one pipeline.

```
Jolpica + FastF1 ──► ingest ──► pace_model ──► features ──► position_model (walk-forward CV, SHAP)
                                    │                                   
                                    └──► per-circuit tyre deg / fuel / compound offsets ──► simulator ──► dashboard
```

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=src            # Windows PowerShell: $env:PYTHONPATH="src"

# Option A: offline, synthetic data (validates the whole pipeline)
python -m f1lab.synthetic

# Option B: real data (needs internet; first run downloads a lot, FastF1 caches it)
python -m f1lab.ingest --years 2022 2023 2024 2025

python -m f1lab.pipeline         # features, models, metrics -> data/processed/
streamlit run app/dashboard.py
pytest
```

## What each module does

| Module | Purpose |
|---|---|
| `ingest.py` | Results + qualifying (Jolpica), laps/tyres/weather (FastF1) -> Parquet |
| `synthetic.py` | Fake data with the same schema, with known tyre/fuel parameters so models can be checked against truth |
| `pace_model.py` | Per-race regression that strips tyre age, compound and fuel load out of lap times, giving each driver's underlying race pace plus per-circuit degradation |
| `features.py` | Leakage-safe features (everything rolling is shifted one race) |
| `position_model.py` | LightGBM finishing-position and podium models, walk-forward validation vs a grid-order baseline, SHAP importance |
| `simulator.py` | Vectorised Monte Carlo: shared safety-car windows, pit loss (discounted under SC), DNFs, strategy enumeration and comparison |
| `app/dashboard.py` | Interactive dashboard: overview, race predictor (timing tower), strategy lab (optimiser + build-your-own plan), race replay, pace and tyres, head to head |
| `app/theme.py` | F1-inspired dark theme, timing-tower component, tyre colours |

## Honest notes on results

* On the **synthetic** data the position model only ties the grid-order baseline. That is expected: synthetic finishing order is almost fully set by pace plus qualifying noise, so there is little signal beyond grid. The pace model does recover the true tyre degradation rates, which is the part synthetic data can verify. Judge the prediction model on **real** data.
* The model predicts the shift from grid slot (not the raw position) and uses small, heavily regularised trees, because ~20 rows per race is little data.
* Walk-forward validation only ever trains on earlier races. Compare against the grid baseline, not against zero.

## Known simplifications and next steps

1. **Traffic / dirty air** is not modelled in the simulator (rivals are independent). Adding a gap-dependent penalty would make undercut/overcut effects appear.
2. **Rain** laps are excluded from pace fitting; wet-race strategy is out of scope.
3. **Safety-car probability** is a slider; estimate it per circuit from historical `TrackStatus` codes.
4. **Real grid:** `scenario_from_data` orders the grid by recent pace; swap in actual qualifying.
5. Add **driver-vs-car decomposition** (idea #2) using the `pace_delta` table, e.g. a mixed-effects model with teammate comparison.
6. Reliability (DNF) is a flat probability; model it per team from `status`.
7. FastF1 lap `Driver` is a 3-letter code; results carry `driver_code` from Jolpica so the two join.

## Dashboard

* **Overview:** model KPIs vs grid baseline, championship lines, constructors, SHAP importance.
* **Race predictor:** timing tower for any out-of-fold race (predicted vs actual, podium probability, places vs grid).
* **Strategy lab:** one-click optimiser over every legal 1-2 stop plan, plus a builder for your own plan with a tyre-strategy timeline, finishing-position distribution and pace trace.
* **Race replay:** simulate a full race from a seed, scrub lap by lap, highlight drivers, see safety-car period and retirements.
* **Pace & tyres:** corrected driver pace, tyre wear curves, per-circuit degradation, driver head-to-head.

Add your own images (logo, header banner, team crests) in `app/assets/`; see the README there.
