"""Original F1-inspired look: carbon-fibre dark theme, timing tower, tyre colours.

No official logos or photos are used; everything is drawn with CSS / SVG / Plotly.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

RED = "#e10600"
BG = "#0b0b0f"
PANEL = "#15151c"
TEXT = "#f2f2f5"
MUTED = "#9a9aa8"

TYRE_COLORS = {"SOFT": "#ff2d2d", "MEDIUM": "#ffd12e", "HARD": "#f2f2f2"}

# approximate livery colours for real Jolpica constructor ids
TEAM_COLORS = {
    "red_bull": "#3671c6", "ferrari": "#e8002d", "mercedes": "#27f4d2", "mclaren": "#ff8000",
    "aston_martin": "#229971", "alpine": "#ff87bc", "williams": "#64c4ff", "rb": "#6692ff",
    "haas": "#b6babd", "sauber": "#52e252", "alphatauri": "#5e8faa", "alfa": "#c92d4b",
    "audi": "#d0d0d0", "cadillac": "#9aa0a6",
}
_FALLBACK = ["#e8002d", "#3671c6", "#27f4d2", "#ff8000", "#229971", "#ff87bc",
             "#64c4ff", "#b6babd", "#52e252", "#c792ea", "#ffd12e", "#8bd450"]


def team_color(constructor_id: str, order: list | None = None) -> str:
    if constructor_id in TEAM_COLORS:
        return TEAM_COLORS[constructor_id]
    idx = (order.index(constructor_id) if order and constructor_id in order else hash(constructor_id)) % len(_FALLBACK)
    return _FALLBACK[idx]


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Titillium Web', sans-serif; }}
.stApp {{
  background:
    repeating-linear-gradient(45deg, rgba(255,255,255,0.018) 0 2px, transparent 2px 6px),
    repeating-linear-gradient(-45deg, rgba(255,255,255,0.018) 0 2px, transparent 2px 6px),
    {BG};
  color: {TEXT};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 2px solid {RED}; }}
.block-container {{ padding-top: 1.2rem; max-width: 1250px; }}

.hero {{
  position: relative; overflow: hidden; border-radius: 6px; padding: 22px 28px 20px 28px; margin-bottom: 14px;
  background: linear-gradient(100deg, #1c1c25 0%, #101016 60%, #1a0a0a 100%);
  border: 1px solid #262633;
}}
.hero:before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:8px; background:{RED}; }}
.hero:after {{
  content:""; position:absolute; right:0; top:0; bottom:0; width:220px; opacity:.22;
  background: conic-gradient(#fff 25%, #000 0 50%, #fff 0 75%, #000 0) 0 0/28px 28px;
  -webkit-mask-image: linear-gradient(90deg, transparent, #000); mask-image: linear-gradient(90deg, transparent, #000);
}}
.hero h1 {{ margin:0; font-weight:900; letter-spacing:.5px; font-size:2.1rem; text-transform:uppercase; font-style:italic; }}
.hero h1 span {{ color:{RED}; }}
.hero p {{ margin:2px 0 0 0; color:{MUTED}; font-size:.95rem; letter-spacing:.4px; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid #262633; }}
.stTabs [data-baseweb="tab"] {{
  background: {PANEL}; border-radius: 4px 4px 0 0; padding: 8px 18px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .6px; font-size: .85rem; color: {MUTED};
}}
.stTabs [aria-selected="true"] {{ background: {RED}; color: #fff; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: transparent; }}

.kpi {{ background:{PANEL}; border:1px solid #262633; border-top:3px solid {RED}; border-radius:4px; padding:12px 16px; }}
.kpi .l {{ color:{MUTED}; font-size:.72rem; text-transform:uppercase; letter-spacing:1px; font-weight:600; }}
.kpi .v {{ font-size:1.9rem; font-weight:900; line-height:1.1; font-style:italic; }}
.kpi .s {{ color:{MUTED}; font-size:.78rem; }}
.good {{ color:#3ddc84; }} .bad {{ color:#ff5a5a; }}

.tower {{ background:{PANEL}; border:1px solid #262633; border-radius:4px; overflow:hidden; }}
.tower .hd, .tower .row {{ display:grid; grid-template-columns: 38px 8px 70px 1fr 62px 62px 70px; gap:8px; align-items:center; padding:5px 10px; }}
.tower .hd {{ background:#0e0e13; color:{MUTED}; font-size:.68rem; text-transform:uppercase; letter-spacing:1px; font-weight:700; }}
.tower .row {{ border-top:1px solid #1f1f29; font-size:.92rem; }}
.tower .row:nth-child(even) {{ background:#12121a; }}
.tower .pos {{ background:#fff; color:#111; font-weight:900; text-align:center; border-radius:3px; padding:1px 0; font-style:italic; }}
.tower .bar {{ width:6px; height:22px; border-radius:2px; }}
.tower .code {{ font-weight:900; letter-spacing:1px; }}
.tower .meter {{ height:9px; background:#23232e; border-radius:5px; overflow:hidden; }}
.tower .meter > div {{ height:100%; background:linear-gradient(90deg,#7a0300,{RED}); }}
.tower .num {{ text-align:right; font-variant-numeric: tabular-nums; }}
.tower .up {{ color:#3ddc84; font-weight:700; }} .tower .dn {{ color:#ff5a5a; font-weight:700; }} .tower .eq {{ color:{MUTED}; }}
.tyre {{ display:inline-block; width:14px; height:14px; border-radius:50%; border:3px solid; box-sizing:border-box; vertical-align:middle; margin-right:5px; }}

.sec {{ display:flex; align-items:center; gap:10px; margin:14px 0 6px 0; font-weight:900; text-transform:uppercase; letter-spacing:1px; font-style:italic; }}
.sec:before {{ content:""; width:26px; height:6px; background:{RED}; transform:skewX(-30deg); }}
.stripe {{ height:4px; background:linear-gradient(90deg,{RED} 0 30%, #fff 30% 33%, {RED} 33% 36%, transparent 36%); margin:6px 0 14px 0; }}
.stButton > button {{ background:{RED}; color:#fff; border:none; font-weight:800; text-transform:uppercase; letter-spacing:.8px; border-radius:3px; }}
.stButton > button:hover {{ background:#ff2a1f; color:#fff; }}
</style>
"""

def hero_html(logo_uri: str | None = None, bg_uri: str | None = None) -> str:
    """Header banner. `logo_uri` / `bg_uri` are optional data-URIs of images the user supplied."""
    bg = (f'background-image:linear-gradient(100deg,rgba(11,11,15,.92),rgba(11,11,15,.55)),url({bg_uri});'
          'background-size:cover;background-position:center;') if bg_uri else ""
    logo = f'<img src="{logo_uri}" style="height:54px;margin-right:18px;object-fit:contain"/>' if logo_uri else ""
    return (f'<div class="hero" style="{bg}"><div style="display:flex;align-items:center">{logo}<div>'
            '<h1>F1 <span>Strategy</span> Lab</h1>'
            '<p>Race-outcome prediction &nbsp;·&nbsp; Monte Carlo pit-strategy simulation &nbsp;·&nbsp; Race replay</p>'
            '</div></div></div>')


def section(title: str) -> str:
    return f'<div class="sec">{title}</div>'


def kpi(label: str, value: str, sub: str = "", cls: str = "") -> str:
    return f'<div class="kpi"><div class="l">{label}</div><div class="v {cls}">{value}</div><div class="s">{sub}</div></div>'


def tyre_dot(compound: str) -> str:
    return f'<span class="tyre" style="border-color:{TYRE_COLORS.get(compound, "#888")}"></span>'


def tower_html(rows: list, headers=("Pos", "", "Driver", "", "", "", ""), cols=("Δ", "Info", "Result")) -> str:
    """rows: dicts with pos, code, color, meter (0-1), delta (int|None), info, result, extra_html."""
    head = (f'<div class="hd"><div>{headers[0]}</div><div></div><div>Driver</div><div></div>'
            f'<div class="num">{cols[0]}</div><div class="num">{cols[1]}</div><div class="num">{cols[2]}</div></div>')
    body = []
    for r in rows:
        d = r.get("delta")
        if d is None:
            dh = '<span class="eq">–</span>'
        elif d > 0:
            dh = f'<span class="up">▲{d}</span>'
        elif d < 0:
            dh = f'<span class="dn">▼{abs(d)}</span>'
        else:
            dh = '<span class="eq">=</span>'
        meter = f'<div class="meter"><div style="width:{int(100 * r.get("meter", 0))}%"></div></div>' if "meter" in r else ""
        body.append(
            f'<div class="row"><div class="pos">{r["pos"]}</div><div class="bar" style="background:{r["color"]}"></div>'
            f'<div class="code">{r.get("extra_html", "")}{r["code"]}</div><div>{meter}</div>'
            f'<div class="num">{dh}</div><div class="num">{r.get("info", "")}</div><div class="num">{r.get("result", "")}</div></div>'
        )
    return f'<div class="tower">{head}{"".join(body)}</div>'


def register_plotly_template() -> None:
    pio.templates["f1"] = go.layout.Template(
        layout=dict(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(21,21,28,0.65)",
            font=dict(family="Titillium Web, sans-serif", color=TEXT),
            colorway=[RED, "#27f4d2", "#ffd12e", "#3671c6", "#ff8000", "#ff87bc", "#52e252"],
            xaxis=dict(gridcolor="#262633", zerolinecolor="#262633"),
            yaxis=dict(gridcolor="#262633", zerolinecolor="#262633"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=40, r=20, t=50, b=40),
            hoverlabel=dict(bgcolor=PANEL, font_family="Titillium Web"),
        )
    )
    pio.templates.default = "f1"
