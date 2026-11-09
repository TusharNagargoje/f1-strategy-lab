"""F1 Strategy Lab dashboard.

    streamlit run app/dashboard.py        (from the repo root, after `python -m f1lab.pipeline`)

Optional: put your own images in app/assets/  (logo.png, hero.jpg, teams/<constructor_id>.png).
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import theme
from f1lab.config import PROC_DIR
from f1lab.pipeline import scenario_from_data
from f1lab.simulator import (Driver, Strategy, compare_strategies, enumerate_strategies,
                             expected_lap_trace, race_trace, random_strategy, simulate_race,
                             stint_segments)

st.set_page_config(page_title="F1 Strategy Lab", page_icon="🏁", layout="wide")
theme.register_plotly_template()
st.markdown(theme.CSS, unsafe_allow_html=True)

ASSETS = ROOT / "app" / "assets"
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".svg": "image/svg+xml"}


@st.cache_data
def asset_uri(stem: str, folder: str = "") -> str | None:
    """Data-URI of a user-supplied image app/assets/<folder>/<stem>.<ext>, or None."""
    base = ASSETS / folder if folder else ASSETS
    for ext, mime in _MIME.items():
        f = base / f"{stem}{ext}"
        if f.exists():
            return f"data:{mime};base64," + base64.b64encode(f.read_bytes()).decode()
    return None


st.markdown(theme.hero_html(asset_uri("logo"), asset_uri("hero")), unsafe_allow_html=True)

if not (PROC_DIR / "metrics.json").exists():
    st.error("No processed data yet. Run `python -m f1lab.synthetic` (or `python -m f1lab.ingest ...`) "
             "and then `python -m f1lab.pipeline`, and reload this page.")
    st.stop()


@st.cache_data
def load():
    feats = pd.read_parquet(PROC_DIR / "features.parquet")
    if "driver_code" not in feats:
        feats["driver_code"] = feats["driver_id"]
    return dict(
        feats=feats,
        pace=pd.read_parquet(PROC_DIR / "pace.parquet"),
        comps=pd.read_parquet(PROC_DIR / "compounds.parquet"),
        oof=pd.read_parquet(PROC_DIR / "oof_predictions.parquet"),
        imp=pd.read_csv(PROC_DIR / "feature_importance.csv"),
        metrics=json.load(open(PROC_DIR / "metrics.json")),
    )


D = load()
feats, pace, comps, oof, imp, metrics = D["feats"], D["pace"], D["comps"], D["oof"], D["imp"], D["metrics"]

latest = feats.sort_values("race_idx").groupby("driver_code").tail(1)
TEAM_OF = dict(zip(latest["driver_code"], latest["constructor_id"]))
TEAM_ORDER = sorted(set(feats["constructor_id"]))
CODE_OF = dict(zip(feats["driver_id"], feats["driver_code"]))
RACE_LABEL = {
    int(r.race_idx): f"{int(r.season)} · R{int(r.round)} · {r.circuit_id}"
    for r in feats.drop_duplicates("race_idx").itertuples()
}


def color_of(code: str) -> str:
    return theme.team_color(TEAM_OF.get(code, code), TEAM_ORDER)


def team_logo(code: str) -> str:
    uri = asset_uri(TEAM_OF.get(code, ""), "teams")
    return f'<img src="{uri}" style="height:16px;margin-right:6px;vertical-align:middle"/>' if uri else ""


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(theme.section("Race setup"), unsafe_allow_html=True)
    circuits = sorted(comps["circuit"].unique())
    circuit = st.selectbox("Circuit", circuits)
    sc_prob = st.slider("Safety-car probability", 0.0, 1.0, 0.45, 0.05)
    pit_loss = st.slider("Pit-lane time loss (s)", 15.0, 30.0, 21.0, 0.5)
    form_n = st.slider("Driver form window (races)", 1, 10, 5)
    sims = st.select_slider("Simulations", [500, 1000, 2000, 5000], value=1000)
    st.markdown(theme.section("Filters"), unsafe_allow_html=True)
    seasons = sorted(feats["season"].unique())
    sel_seasons = st.multiselect("Seasons", seasons, default=seasons)
    with st.expander("Add your own images"):
        st.caption("Drop files into `app/assets/`: `logo.png`, `hero.jpg`, and team crests as "
                   "`teams/<constructor_id>.png`. They appear automatically after a reload.")

params, field = scenario_from_data(circuit, pit_loss=pit_loss, sc_prob=sc_prob, last_n_races=form_n)
names = [d.name for d in field]
with st.sidebar:
    target = st.selectbox("Driver to optimise", names, index=min(4, len(names) - 1))
t_idx = names.index(target)

tab_over, tab_pred, tab_lab, tab_replay, tab_pace = st.tabs(
    ["Overview", "Race predictor", "Strategy lab", "Race replay", "Pace & tyres"])

# ----------------------------------------------------------------- overview
with tab_over:
    m, b = metrics["model"], metrics["grid_baseline"]
    c = st.columns(4)
    c[0].markdown(theme.kpi("Held-out races", str(metrics["n_test_races"]), "walk-forward validation"), unsafe_allow_html=True)
    c[1].markdown(theme.kpi("Position MAE", f"{m['mae']:.2f}", f"grid baseline {b['mae']:.2f}",
                            "good" if m["mae"] <= b["mae"] else "bad"), unsafe_allow_html=True)
    c[2].markdown(theme.kpi("Rank correlation", f"{m['spearman']:.2f}", f"grid baseline {b['spearman']:.2f}",
                            "good" if m["spearman"] >= b["spearman"] else "bad"), unsafe_allow_html=True)
    c[3].markdown(theme.kpi("Podium AUC", f"{m.get('podium_auc', float('nan')):.3f}",
                            f"grid baseline {b.get('podium_auc', float('nan')):.3f}"), unsafe_allow_html=True)

    st.markdown(theme.section("Championship"), unsafe_allow_html=True)
    season = st.selectbox("Season", sel_seasons or seasons, index=len(sel_seasons or seasons) - 1)
    s = feats[feats["season"] == season].sort_values("round")
    s = s.assign(cum=s.groupby("driver_code")["points"].cumsum())
    totals = s.groupby("driver_code")["cum"].max().sort_values(ascending=False)
    chosen = st.multiselect("Drivers", list(totals.index), default=list(totals.index[:6]))
    fig = go.Figure()
    for code in chosen:
        d = s[s["driver_code"] == code]
        fig.add_trace(go.Scatter(x=d["round"], y=d["cum"], mode="lines+markers", name=code,
                                 line=dict(color=color_of(code), width=3), marker=dict(size=6),
                                 hovertemplate=f"<b>{code}</b><br>Round %{{x}}<br>%{{y}} pts<extra></extra>"))
    fig.update_layout(title=f"{season} drivers' points", xaxis_title="Round", yaxis_title="Points", height=380)
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    con = s.groupby("constructor_id")["points"].sum().sort_values().reset_index()
    left.plotly_chart(go.Figure(go.Bar(
        x=con["points"], y=con["constructor_id"], orientation="h",
        marker_color=[theme.team_color(t, TEAM_ORDER) for t in con["constructor_id"]],
        hovertemplate="%{y}: %{x} pts<extra></extra>")).update_layout(title="Constructors", height=340),
        width="stretch")
    ii = imp.sort_values("importance")
    right.plotly_chart(go.Figure(go.Bar(x=ii["importance"], y=ii["feature"], orientation="h",
                                        marker_color=theme.RED)).update_layout(
        title="What drives the prediction (mean |SHAP|)", height=340), width="stretch")

# ---------------------------------------------------------------- predictor
with tab_pred:
    st.markdown(theme.section("Predicted classification"), unsafe_allow_html=True)
    races = sorted(oof["race_idx"].unique())
    c1, c2 = st.columns([3, 1])
    race = c1.selectbox("Race (out-of-fold: model never saw it in training)", races[::-1],
                        format_func=lambda i: RACE_LABEL.get(int(i), str(i)))
    view = c2.radio("Show", ["Predicted", "Actual"], horizontal=True)
    r = oof[oof["race_idx"] == race].copy()
    r["code"] = r["driver_id"].map(CODE_OF)
    r = r.sort_values("pred_position" if view == "Predicted" else "position").reset_index(drop=True)
    pmax = max(r["pred_podium_p"].max(), 1e-9)
    rows = []
    for i, x in r.iterrows():
        rows.append(dict(
            pos=i + 1, code=x["code"], color=color_of(x["code"]), extra_html=team_logo(x["code"]),
            meter=x["pred_podium_p"] / pmax, delta=int(x["grid"] - (i + 1)),
            info=f'{x["pred_podium_p"]:.0%}', result=f'P{int(x["position"])}'))
    hits = len(set(r.nsmallest(3, "pred_position")["code"]) & set(r.nsmallest(3, "position")["code"]))
    left, right = st.columns([3, 2])
    left.markdown(theme.tower_html(rows, cols=("vs grid", "Podium %", "Actual")), unsafe_allow_html=True)
    with right:
        st.markdown(theme.kpi("Podium picks right", f"{hits} / 3",
                              "predicted top-3 vs actual top-3", "good" if hits >= 2 else ""), unsafe_allow_html=True)
        sc = go.Figure()
        sc.add_trace(go.Scatter(x=[1, len(r)], y=[1, len(r)], mode="lines", line=dict(color="#555", dash="dot"),
                                hoverinfo="skip", showlegend=False))
        sc.add_trace(go.Scatter(
            x=r["pred_position"], y=r["position"], mode="markers+text", text=r["code"], textposition="top center",
            marker=dict(size=11, color=[color_of(c) for c in r["code"]], line=dict(color="#fff", width=1)),
            hovertemplate="<b>%{text}</b><br>predicted %{x:.1f}<br>actual %{y}<extra></extra>", showlegend=False))
        sc.update_layout(title="Predicted vs actual finish", xaxis_title="Predicted position",
                         yaxis_title="Actual position", height=470, yaxis_autorange="reversed")
        st.plotly_chart(sc, width="stretch")

# ------------------------------------------------------------- strategy lab
with tab_lab:
    st.markdown(theme.section(f"Strategy lab · {target} · {circuit}"), unsafe_allow_html=True)
    st.caption(f"{params.n_laps} laps · base lap {params.base_lap:.1f}s · "
               + " · ".join(f"{k} {v:.3f}s/lap" for k, v in params.deg.items())
               + f" · grid P{field[t_idx].grid}")

    key = (circuit, target, sc_prob, pit_loss, sims, form_n)
    if st.button("Find best strategy", key="optimise"):
        with st.spinner("Simulating hundreds of strategies..."):
            table = compare_strategies(params, field, target, enumerate_strategies(params), sims=max(300, sims // 3))
        st.session_state["opt"] = dict(key=key, table=table)
    opt = st.session_state.get("opt")
    opt_ok = bool(opt and opt["key"] == key)

    st.markdown(theme.section("Build your own plan"), unsafe_allow_html=True)
    stops = st.radio("Pit stops", [1, 2, 3], horizontal=True, key="stops")
    defaults = ["MEDIUM", "HARD", "MEDIUM", "HARD"]
    cc = st.columns(stops + 1)
    comp_sel = [cc[i].selectbox(f"Stint {i + 1} tyre", COMPOUNDS, index=COMPOUNDS.index(defaults[i]),
                                key=f"comp_{i}") for i in range(stops + 1)]
    pc = st.columns(stops)
    pit_sel = [pc[i].slider(f"Pit {i + 1} on lap", 6, params.n_laps - 6,
                            int(round(params.n_laps * (i + 1) / (stops + 1))),
                            key=f"pit_{stops}_{params.n_laps}_{i}") for i in range(stops)]
    problems = []
    if len(set(comp_sel)) < 2:
        problems.append("Use at least two different tyre compounds.")
    if any(b - a < 6 for a, b in zip(pit_sel, pit_sel[1:])) or pit_sel != sorted(pit_sel):
        problems.append("Pit stops must be in order and at least 6 laps apart.")
    if problems:
        st.warning(" ".join(problems))
        st.session_state.pop("custom", None)
    else:
        custom = Strategy(comp_sel, pit_sel)
        st.session_state["custom"] = dict(strategy=custom, target=target)
        plans = {"Your plan": custom, "Baseline": Strategy(["MEDIUM", "HARD"], [params.n_laps // 2])}
        if opt_ok:
            top = opt["table"].iloc[0]
            best_label = top["strategy"]
            tr_row = enumerate_strategies(params)
            best = next((s for s in tr_row if s.label() == best_label), None)
            if best:
                plans["Optimiser best"] = best

        def run(strategy):
            ds = list(field)
            ds[t_idx] = Driver(target, field[t_idx].pace_delta, field[t_idx].grid, strategy)
            pos, times = simulate_race(params, ds, sims=sims, seed=1)
            return pos[:, t_idx]

        results = {k: run(v) for k, v in plans.items()}
        k = st.columns(len(plans))
        for col, (name, p) in zip(k, results.items()):
            col.markdown(theme.kpi(name, f"P{p.mean():.1f}", f"{plans[name].label()} · podium {(p <= 3).mean():.0%}"
                                   f" · points {(p <= 10).mean():.0%}"), unsafe_allow_html=True)

        # stint timeline (TV-style)
        tl = go.Figure()
        seen = set()
        for row, (name, strat) in enumerate(plans.items()):
            for comp, a, b in stint_segments(params, strat):
                tl.add_trace(go.Bar(
                    y=[name], x=[b - a + 1], base=[a - 1], orientation="h", name=comp,
                    legendgroup=comp, showlegend=comp not in seen,
                    marker=dict(color=theme.TYRE_COLORS[comp], line=dict(color="#0b0b0f", width=2)),
                    text=f"{comp[0]} · {b - a + 1}", textposition="inside",
                    textfont=dict(color="#111" if comp != "SOFT" else "#fff", size=13),
                    hovertemplate=f"{name}<br>{comp}: laps {a}-{b}<extra></extra>"))
                seen.add(comp)
        tl.update_layout(barmode="overlay", title="Tyre strategy", height=120 + 55 * len(plans),
                         xaxis=dict(range=[0, params.n_laps], title="Lap"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(tl, width="stretch")

        a, b_ = st.columns(2)
        dist = go.Figure()
        xs = np.arange(1, len(field) + 1)
        for name, p in results.items():
            dist.add_trace(go.Bar(x=xs, y=[(p == x).mean() for x in xs], name=name))
        dist.update_layout(barmode="group", title="Finishing-position probability",
                           xaxis_title="Finish position", yaxis_title="Probability", yaxis_tickformat=".0%", height=360)
        a.plotly_chart(dist, width="stretch")

        lt = go.Figure()
        for name, strat in plans.items():
            d = Driver(target, field[t_idx].pace_delta, field[t_idx].grid, strat)
            y = expected_lap_trace(params, d)
            for lap in strat.pit_laps:
                y[lap - 1] -= params.pit_loss
            y[0] = np.nan
            lt.add_trace(go.Scatter(x=np.arange(1, params.n_laps + 1), y=y, mode="lines", name=name,
                                    hovertemplate="Lap %{x}: %{y:.2f}s<extra>" + name + "</extra>"))
        lt.update_layout(title="On-track pace by lap (pit loss removed)", xaxis_title="Lap",
                         yaxis_title="Lap time (s)", height=360)
        b_.plotly_chart(lt, width="stretch")

    if opt_ok:
        st.markdown(theme.section("Optimiser: top strategies"), unsafe_allow_html=True)
        tbl = opt["table"].head(12).copy()
        tbl["podium_p"] = (tbl["podium_p"] * 100).round(1)
        tbl["points_p"] = (tbl["points_p"] * 100).round(1)
        st.dataframe(tbl.round(2), hide_index=True, width="stretch",
                     column_config={"podium_p": "podium %", "points_p": "points %"})
        st.plotly_chart(px.scatter(opt["table"], x="mean_time", y="mean_pos", color=opt["table"]["stops"].astype(str),
                                   hover_name="strategy", labels={"color": "stops"},
                                   title="Every strategy tried (lower-left is better)"), width="stretch")
    else:
        st.info("Click **Find best strategy** to search every legal one- and two-stop plan for this driver.")

# ------------------------------------------------------------------- replay
with tab_replay:
    st.markdown(theme.section(f"Race replay · {circuit}"), unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3])
    seed = c1.number_input("Race seed", 0, 9999, 7, help="Same seed = same race. Change it to roll a new one.")
    if c1.button("Simulate race", key="replay_btn"):
        rng = np.random.default_rng(int(seed))
        ds = []
        for d in field:
            ds.append(Driver(d.name, d.pace_delta, d.grid, random_strategy(params, rng)))
        cust = st.session_state.get("custom")
        if cust and cust["target"] == target:
            ds[t_idx] = Driver(target, field[t_idx].pace_delta, field[t_idx].grid, cust["strategy"])
        st.session_state["replay"] = dict(
            key=(circuit, sc_prob, pit_loss, form_n, int(seed), target), drivers=ds,
            trace=race_trace(params, ds, seed=int(seed)))
    rp = st.session_state.get("replay")
    if not rp or rp["key"] != (circuit, sc_prob, pit_loss, form_n, int(seed), target):
        c2.info("Set a seed and press **Simulate race**. Your driver uses the plan from the Strategy lab; "
                "everyone else gets a random plausible strategy.")
    else:
        tr, ds = rp["trace"], rp["drivers"]
        n = params.n_laps
        highlight = c2.multiselect("Highlight drivers", names, default=[target])
        lap = st.slider("Lap", 1, n, n)
        fig = go.Figure()
        for i, d in enumerate(ds):
            hl = d.name in highlight
            fig.add_trace(go.Scatter(
                x=np.arange(1, n + 1), y=tr["pos"][i], mode="lines", name=d.name,
                line=dict(color=color_of(d.name), width=4 if hl else 1.5), opacity=1.0 if hl else 0.35,
                hovertemplate=f"<b>{d.name}</b> lap %{{x}}: P%{{y}}<extra></extra>"))
        sc_laps = np.flatnonzero(tr["sc"])
        if len(sc_laps):
            fig.add_vrect(x0=sc_laps.min() + 0.5, x1=sc_laps.max() + 1.5, fillcolor="#ffd12e", opacity=0.12,
                          line_width=0, annotation_text="Safety car", annotation_position="top left")
        fig.add_vline(x=lap, line_color=theme.RED, line_width=2)
        fig.update_layout(title="Race trace (position by lap)", xaxis_title="Lap", yaxis_title="Position",
                          yaxis=dict(autorange="reversed", dtick=1), height=470, showlegend=False)
        st.plotly_chart(fig, width="stretch")

        col = lap - 1
        alive = [i for i in range(len(ds)) if np.isfinite(tr["cum"][i, col])]
        order = sorted(alive, key=lambda i: tr["pos"][i, col])
        lead = tr["cum"][order[0], col]
        rows = []
        for i in order:
            d = ds[i]
            seg = next(s for s in stint_segments(params, d.strategy) if s[1] <= lap <= s[2])
            gap = "Leader" if i == order[0] else f"+{tr['cum'][i, col] - lead:.1f}s"
            rows.append(dict(pos=int(tr["pos"][i, col]), code=d.name, color=color_of(d.name),
                             extra_html=theme.tyre_dot(seg[0]) + team_logo(d.name),
                             delta=int(d.grid - tr["pos"][i, col]), info=gap, result=f"{seg[0][0]}{lap - seg[1] + 1}"))
        for i in range(len(ds)):
            if i not in alive:
                rows.append(dict(pos="–", code=ds[i].name, color=color_of(ds[i].name), delta=None,
                                 info="OUT", result=f"L{int(tr['dnf_lap'][i])}"))
        st.markdown(theme.tower_html(rows, cols=("vs grid", "Gap", "Tyre age")), unsafe_allow_html=True)

# --------------------------------------------------------------- pace & tyres
with tab_pace:
    st.markdown(theme.section("Underlying race pace"), unsafe_allow_html=True)
    pf = pace[pace["season"].isin(sel_seasons or seasons)]
    avg = pf.groupby("driver_code")["pace_delta"].mean().sort_values().reset_index()
    pfig = go.Figure(go.Bar(
        x=avg["driver_code"], y=avg["pace_delta"], marker_color=[color_of(c) for c in avg["driver_code"]],
        hovertemplate="<b>%{x}</b><br>%{y:+.3f} s/lap vs field median<extra></extra>"))
    pfig.update_layout(title="Fuel- and tyre-corrected pace vs field median (lower = faster)",
                       yaxis_title="s / lap", height=380)
    st.plotly_chart(pfig, width="stretch")

    a, b = st.columns(2)
    prof = pd.DataFrame({"compound": COMPOUNDS, "deg": [params.deg[c] for c in COMPOUNDS],
                         "off": [params.offset[c] for c in COMPOUNDS]})
    ages = np.arange(0, 41)
    dg = go.Figure()
    for _, rw in prof.iterrows():
        dg.add_trace(go.Scatter(x=ages, y=rw["off"] + rw["deg"] * ages, mode="lines", name=rw["compound"],
                                line=dict(color=theme.TYRE_COLORS[rw["compound"]], width=4),
                                hovertemplate=f"{rw['compound']} age %{{x}}: +%{{y:.2f}}s<extra></extra>"))
    dg.update_layout(title=f"Tyre wear curves · {circuit}", xaxis_title="Tyre age (laps)",
                     yaxis_title="Lap-time penalty vs fresh soft (s)", height=380)
    a.plotly_chart(dg, width="stretch")
    cd = comps.groupby(["circuit", "compound"], as_index=False)["deg_per_lap"].mean()
    b.plotly_chart(px.bar(cd, x="circuit", y="deg_per_lap", color="compound", barmode="group",
                          color_discrete_map=theme.TYRE_COLORS, title="Degradation by circuit (s per lap of age)"),
                   width="stretch")

    st.markdown(theme.section("Head to head"), unsafe_allow_html=True)
    pool = sorted(pf["driver_code"].unique())
    h1, h2 = st.columns(2)
    d1 = h1.selectbox("Driver A", pool, index=0)
    d2 = h2.selectbox("Driver B", pool, index=min(1, len(pool) - 1))
    both = pf[pf["driver_code"].isin([d1, d2])].copy()
    both["race"] = both["season"].astype(str) + " R" + both["round"].astype(str)
    hh = go.Figure()
    for dcode in (d1, d2):
        x = both[both["driver_code"] == dcode].sort_values(["season", "round"])
        hh.add_trace(go.Scatter(x=x["race"], y=x["pace_delta"], mode="lines+markers", name=dcode,
                                line=dict(color=color_of(dcode), width=3)))
    hh.update_layout(title="Race pace per round (lower = faster)", yaxis_title="s / lap vs median", height=360,
                     xaxis=dict(showticklabels=False))
    st.plotly_chart(hh, width="stretch")
    piv = both.pivot_table(index=["season", "round"], columns="driver_code", values="pace_delta").dropna()
    if len(piv) and d1 != d2:
        gap = (piv[d1] - piv[d2]).mean()
        wins = (piv[d1] < piv[d2]).mean()
        k1, k2, k3 = st.columns(3)
        k1.markdown(theme.kpi("Average gap", f"{abs(gap):.3f}s", f"{d1 if gap < 0 else d2} faster per lap"), unsafe_allow_html=True)
        k2.markdown(theme.kpi(f"{d1} faster in", f"{wins:.0%}", f"of {len(piv)} shared races"), unsafe_allow_html=True)
        k3.markdown(theme.kpi(f"{d2} faster in", f"{1 - wins:.0%}", f"of {len(piv)} shared races"), unsafe_allow_html=True)
