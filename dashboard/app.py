"""S&P 500 min / max price dashboard.

Run from the repo root (after `python process_data.py` and `python run_range_models.py`):
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp500.candles import TIMEFRAMES, resample_candles  # noqa: E402
from sp500.patterns import PATTERNS, detect_patterns  # noqa: E402
from sp500.range_models import MODEL_NAMES  # noqa: E402

PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"

# Colours (validated categorical palette, slots in fixed order). Candles stay neutral so the
# forecast bands carry the colour: max = slot 1 (blue), min = slot 2 (orange).
C_MAX, C_MIN = "#257de0", "#f47230"
C_UP_CANDLE, C_DN_CANDLE = "#9c9c9c", "#3d3d3d"
MODEL_COLORS = {"hist_quantile": "#b0b0b0", "vol_band": "#7a7a7a", "xgboost": "#257de0",
                "catboost": "#f47230", "lstm": "#00b894", "ensemble": "#6a4ce2"}
PAT_COLORS = {"bullish": "#008a17", "bearish": "#ec4647", "neutral": "#666666"}
PAT_SYMBOL = {"bullish": "triangle-up", "bearish": "triangle-down", "neutral": "diamond"}
MODEL_ORDER = ["ensemble", "xgboost", "catboost", "lstm", "vol_band", "hist_quantile"]

st.set_page_config(page_title="S&P 500 price range forecast", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- data
@st.cache_data
def load():
    need = [PROC / "range_table.parquet", PROC / "range_predictions.parquet",
            PROC / "range_forecast.json", REP / "range_results.csv"]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        return None, missing
    table = pd.read_parquet(need[0])
    pred = pd.read_parquet(need[1])
    fc = json.loads(need[2].read_text())
    res = pd.read_csv(need[3])
    imp = pd.read_csv(REP / "range_feature_importance.csv") if (REP / "range_feature_importance.csv").exists() else None
    pst = pd.read_csv(REP / "range_pattern_stats.csv") if (REP / "range_pattern_stats.csv").exists() else None
    return (table, pred, fc, res, imp, pst), []


data, missing = load()
if data is None:
    st.error("Missing outputs: " + ", ".join(missing))
    st.code("python process_data.py\npython run_range_models.py", language="bash")
    st.stop()
table, pred, fc, res, imp, pst = data
close = fc["close"]
as_of = pd.Timestamp(fc["as_of"])

# --------------------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
h = st.sidebar.radio("Forecast horizon", [5, 21], format_func=lambda x: f"Next {x} trading days "
                     f"({'1 week' if x == 5 else '1 month'})")
models_avail = [m for m in MODEL_ORDER if m in fc["horizons"][str(h)]]
model = st.sidebar.selectbox("Model", models_avail, format_func=lambda m: MODEL_NAMES.get(m, m))
tf = st.sidebar.radio("Candle timeframe", list(TIMEFRAMES), format_func=TIMEFRAMES.get, horizontal=True,
                      help="Daily candles are re-sampled into weekly / monthly candles (open = first, "
                           "high = max, low = min, close = last).")
window = st.sidebar.select_slider("History shown", ["3M", "6M", "1Y", "2Y", "5Y", "10Y"], value="6M")
show_pat = st.sidebar.checkbox("Mark candlestick patterns", value=True)
st.sidebar.caption(f"Data through {as_of:%d %b %Y}. Prices: S&P 500 index (^GSPC).")


def q_prices(m: str, horizon: int):
    d = fc["horizons"][str(horizon)][m]
    return [close * np.exp(v) for v in d["up"]], [close * np.exp(v) for v in d["down"]]


# --------------------------------------------------------------------------- header + KPIs
st.title("S&P 500: predicted min and max price")
st.caption(f"Forecast made at the close of **{as_of:%A %d %B %Y}** for the next **{h} trading days**. "
           f"Model: **{MODEL_NAMES.get(model, model)}**. Bands are the 10%–90% quantiles "
           "(an 80% interval); the headline value is the median.")

up_q, dn_q = q_prices(model, h)
vb_up, vb_dn = q_prices("vol_band", h)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Last close", f"{close:,.0f}")
k2.metric(f"Predicted max (next {h}d)", f"{up_q[1]:,.0f}", f"{(up_q[1] / close - 1) * 100:+.2f}%")
k2.caption(f"80% band: {up_q[0]:,.0f} – {up_q[2]:,.0f}")
k3.metric(f"Predicted min (next {h}d)", f"{dn_q[1]:,.0f}", f"{(dn_q[1] / close - 1) * 100:+.2f}%")
k3.caption(f"80% band: {dn_q[0]:,.0f} – {dn_q[2]:,.0f}")
k4.metric("Expected range (max − min)", f"{up_q[1] - dn_q[1]:,.0f} pts")
k4.caption(f"{(up_q[1] - dn_q[1]) / close * 100:.2f}% of price · volatility baseline: "
           f"{vb_dn[1]:,.0f} – {vb_up[1]:,.0f}")


# --------------------------------------------------------------------------- chart helpers
def candles_for(start, end, tf_):
    d = table.loc[start:end, ["open", "high", "low", "close"]].dropna()
    return resample_candles(d, tf_) if tf_ != "D" else d


def future_dates(start: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start + pd.Timedelta(days=1), periods=n)


def candle_trace(c: pd.DataFrame, name="S&P 500"):
    return go.Candlestick(
        x=c.index, open=c["open"], high=c["high"], low=c["low"], close=c["close"], name=name,
        increasing=dict(line=dict(color=C_UP_CANDLE, width=1), fillcolor="#ffffff"),
        decreasing=dict(line=dict(color=C_DN_CANDLE, width=1), fillcolor=C_DN_CANDLE),
    )


def add_range_box(fig, x0, x1, uq, dq, label_suffix="", show_text=True):
    for q, col, nm in ((uq, C_MAX, "max"), (dq, C_MIN, "min")):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=q[0], y1=q[2], fillcolor=col, opacity=0.15,
                      line_width=0, layer="below")
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[q[1], q[1]], mode="lines", line=dict(color=col, width=2, dash="dash"),
            name=f"Predicted {nm}{label_suffix}",
            hovertemplate=f"Predicted {nm}: %{{y:,.0f}}<br>80% band {q[0]:,.0f} – {q[2]:,.0f}<extra></extra>"))
        if show_text:
            fig.add_annotation(x=x1, y=q[1], text=f"{nm} {q[1]:,.0f}", showarrow=False, xanchor="left",
                               font=dict(color="#191919", size=12), xshift=4)


def add_patterns(fig, c: pd.DataFrame, max_marks=60):
    if len(c) < 5:
        return
    flags = detect_patterns(c).tail(max_marks)
    pad = (c["high"] - c["low"]).median() * 0.6
    # one trace per direction (bullish / bearish / neutral); pattern names go in the hover text
    for direction in ("bullish", "bearish", "neutral"):
        names = [n for n, (d, _) in PATTERNS.items() if d == direction]
        hits = flags[names].fillna(0)
        hits = hits[hits.sum(axis=1) > 0]
        if hits.empty:
            continue
        text = ["<br>".join(f"<b>{n.replace('_', ' ')}</b>: {PATTERNS[n][1]}" for n in names if r[n] == 1)
                for _, r in hits.iterrows()]
        y = c.loc[hits.index, "low"] - pad if direction == "bullish" else c.loc[hits.index, "high"] + pad
        fig.add_trace(go.Scatter(
            x=hits.index, y=y, mode="markers", name=f"{direction} pattern", text=text,
            marker=dict(symbol=PAT_SYMBOL[direction], size=9, color=PAT_COLORS[direction],
                        line=dict(color="#ffffff", width=1)),
            hovertemplate="%{text}<extra>" + direction + "</extra>"))


def base_layout(fig, height=560):
    fig.update_layout(
        height=height, template="plotly_white", hovermode="x unified", margin=dict(l=10, r=90, t=30, b=10),
        xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.06, x=0),
        yaxis=dict(title="Index level", gridcolor="#eeeeee", tickformat=",.0f"),
        xaxis=dict(gridcolor="#f4f4f4"),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])] if tf == "D" else [])
    return fig


offsets = {"3M": 3, "6M": 6, "1Y": 12, "2Y": 24, "5Y": 60, "10Y": 120}
start = as_of - pd.DateOffset(months=offsets[window])

tab_fc, tab_bt, tab_cmp, tab_pat, tab_how = st.tabs(
    ["Forecast", "Backtest a date", "Model comparison", "Pattern recognition", "Features & method"])

# --------------------------------------------------------------------------- forecast tab
with tab_fc:
    c = candles_for(start, as_of, tf)
    fig = go.Figure([candle_trace(c)])
    fd = future_dates(as_of, h)
    add_range_box(fig, fd[0], fd[-1], up_q, dn_q)
    fig.add_vline(x=as_of, line=dict(color="#999999", width=1, dash="dot"))
    if show_pat:
        add_patterns(fig, c)
    st.plotly_chart(base_layout(fig), width="stretch")
    st.caption("Shaded boxes: 80% band for the highest high (blue) and lowest low (orange) between "
               f"{fd[0]:%d %b} and {fd[-1]:%d %b %Y} (weekdays; exchange holidays not removed). "
               "Dashed line: median forecast.")

    st.subheader("All models for this horizon")
    rows = []
    for m in models_avail:
        u, d = q_prices(m, h)
        rows.append({"Model": MODEL_NAMES.get(m, m), "Max (median)": u[1], "Max 80% band": f"{u[0]:,.0f} – {u[2]:,.0f}",
                     "Min (median)": d[1], "Min 80% band": f"{d[0]:,.0f} – {d[2]:,.0f}",
                     "Range %": (u[1] - d[1]) / close * 100})
    st.dataframe(pd.DataFrame(rows).style.format({"Max (median)": "{:,.0f}", "Min (median)": "{:,.0f}",
                                                  "Range %": "{:.2f}%"}), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- backtest tab
with tab_bt:
    p = pred[(pred["model"] == model) & (pred["h"] == h)]
    dates = p["date"].drop_duplicates().sort_values()
    st.write("Pick any past day: the chart shows what the model predicted **that day** (out of sample: the "
             "model was trained only on earlier data) and what really happened next.")
    pick = st.date_input("Forecast date", value=pd.Timestamp("2025-04-02").date(),
                         min_value=dates.iloc[0].date(), max_value=dates.iloc[-1].date())
    pick = dates[dates <= pd.Timestamp(pick)].iloc[-1]   # snap to the last trading day
    row = p[p["date"] == pick].set_index("target")
    cl = row["close"].iloc[0]
    uq = [cl * np.exp(row.loc["up", k]) for k in ("q10", "q50", "q90")]
    dq = [cl * np.exp(row.loc["down", k]) for k in ("q10", "q50", "q90")]
    pos = table.index.get_loc(pick)
    fut = table.index[pos + 1: pos + 1 + h]
    span = table.iloc[max(pos - 60, 0): pos + h + 15]
    fig = go.Figure([candle_trace(span[["open", "high", "low", "close"]])])
    if len(fut):
        add_range_box(fig, fut[0], fut[-1], uq, dq)
    fig.add_vline(x=pick, line=dict(color="#999999", width=1, dash="dot"))
    fig.update_layout(xaxis=dict(rangebreaks=[dict(bounds=["sat", "mon"])]))
    st.plotly_chart(base_layout(fig, 480), width="stretch")
    if len(fut) == h:
        a_max, a_min = table["high"].iloc[pos + 1: pos + 1 + h].max(), table["low"].iloc[pos + 1: pos + 1 + h].min()
        b1, b2, b3 = st.columns(3)
        b1.metric("Close on forecast day", f"{cl:,.0f}")
        b2.metric("Actual max", f"{a_max:,.0f}", f"error {a_max - uq[1]:+,.0f} pts", delta_color="off")
        b2.caption(("✅ inside" if uq[0] <= a_max <= uq[2] else "❌ outside") + f" the 80% band {uq[0]:,.0f} – {uq[2]:,.0f}")
        b3.metric("Actual min", f"{a_min:,.0f}", f"error {a_min - dq[1]:+,.0f} pts", delta_color="off")
        b3.caption(("✅ inside" if dq[0] <= a_min <= dq[2] else "❌ outside") + f" the 80% band {dq[0]:,.0f} – {dq[2]:,.0f}")
    else:
        st.info("The outcome of this forecast is not known yet.")

    st.subheader("Predicted vs. actual over time")
    yrs = st.slider("Period", int(dates.dt.year.min()), int(dates.dt.year.max()), (2022, int(dates.dt.year.max())))
    pp = p[(p["date"].dt.year >= yrs[0]) & (p["date"].dt.year <= yrs[1])]
    fig2 = go.Figure()
    for t, col, nm in (("up", C_MAX, "max"), ("down", C_MIN, "min")):
        g = pp[pp["target"] == t].sort_values("date")
        fig2.add_trace(go.Scatter(x=g["date"], y=g["q90"] * 100, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig2.add_trace(go.Scatter(x=g["date"], y=g["q10"] * 100, fill="tonexty", fillcolor=col, opacity=0.2,
                                  line=dict(width=0), name=f"80% band ({nm})", hoverinfo="skip"))
        fig2.add_trace(go.Scatter(x=g["date"], y=g["actual"] * 100, mode="lines", line=dict(color=col, width=1.5),
                                  name=f"Actual {nm} move",
                                  hovertemplate=f"Actual {nm}: %{{y:.2f}}%<extra></extra>"))
    fig2.update_layout(height=380, template="plotly_white", hovermode="x unified",
                       yaxis=dict(title=f"Move vs. close, next {h} days (%)", ticksuffix="%"),
                       margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, width="stretch")
    g = pp.dropna(subset=["actual"])
    cov = g.assign(inside=(g["actual"] >= g["q10"]) & (g["actual"] <= g["q90"])).groupby("target")["inside"].mean()
    st.caption(f"Share of days inside the 80% band in this period: max {cov.get('up', np.nan):.0%}, "
               f"min {cov.get('down', np.nan):.0%} (ideal 80%).")

# --------------------------------------------------------------------------- comparison tab
with tab_cmp:
    split = st.radio("Evaluation period", ["walk_forward", "holdout"], horizontal=True,
                     format_func={"walk_forward": "Walk-forward 2000–2021", "holdout": "Holdout 2022 → today"}.get)
    r = res[(res["split"] == split) & (res["h"] == h)].copy()
    r["Model"] = r["model"].map(MODEL_NAMES)
    r["Target"] = r["target"].map({"up": "Max high", "down": "Min low"})
    # improvement of the whole quantile band over the volatility baseline (pinball loss), in %
    ref = r[r["model"] == "vol_band"].set_index("target")["pinball_x100"]
    r["pinball_gain"] = (1 - r["pinball_x100"] / r["target"].map(ref)) * 100
    c1, c2 = st.columns(2)
    for col, metric, title, ref_line, fmt in (
            (c1, "pinball_gain", "Better than the volatility baseline by (%) — higher is better", 0, ".1f"),
            (c2, "coverage_80", "Coverage of the 80% band — ideal 0.80", 0.8, ".2f")):
        fig = go.Figure()
        for m in MODEL_ORDER:
            g = r[r["model"] == m]
            if g.empty or (metric == "pinball_gain" and m == "vol_band"):
                continue
            fig.add_trace(go.Bar(x=g["Target"], y=g[metric], name=MODEL_NAMES[m], marker_color=MODEL_COLORS[m],
                                 marker_line=dict(color="#ffffff", width=2),
                                 hovertemplate=f"{MODEL_NAMES[m]}<br>%{{x}}: %{{y:{fmt}}}<extra></extra>"))
        fig.add_hline(y=ref_line, line=dict(color="#191919", width=1, dash="dot"))
        fig.update_layout(title=title, barmode="group", height=380, template="plotly_white",
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
        col.plotly_chart(fig, width="stretch")
    show = r[["Model", "Target", "n", "mae_pct", "pinball_x100", "coverage_80", "width_pct", "corr",
              "mae_skill_vs_volband"]].sort_values(["Target", "pinball_x100"])
    show.columns = ["Model", "Target", "Days", "MAE %", "Pinball ×100", "Coverage 80%", "Band width %",
                    "Corr", "Skill vs vol band"]
    st.dataframe(show.style.format({"MAE %": "{:.3f}", "Pinball ×100": "{:.3f}", "Coverage 80%": "{:.2f}",
                                    "Band width %": "{:.2f}", "Corr": "{:.2f}", "Skill vs vol band": "{:+.3f}"}),
                 hide_index=True, width="stretch")
    st.caption("MAE %: average error of the median forecast in % of price. Skill > 0 means a smaller MAE than "
               "the volatility-band baseline (today's volatility × a historical multiplier).")

# --------------------------------------------------------------------------- patterns tab
with tab_pat:
    st.write(f"Candlestick patterns detected on **{TIMEFRAMES[tf].lower()}** candles. Patterns are also used "
             "as model features (daily ones directly, weekly / monthly ones through the multi-timeframe "
             "pattern score).")
    c = candles_for(as_of - pd.DateOffset(months=offsets[window]), as_of, tf)
    flags = detect_patterns(candles_for(as_of - pd.DateOffset(years=3), as_of, tf))
    recent = flags.tail(10)
    found = [(d, n) for d in recent.index for n in PATTERNS if recent.loc[d, n] == 1]
    if found:
        st.dataframe(pd.DataFrame([{"Candle": d.date(), "Pattern": n.replace("_", " "),
                                    "Signal": PATTERNS[n][0], "Meaning": PATTERNS[n][1]} for d, n in reversed(found)]),
                     hide_index=True, width="stretch")
    else:
        st.info("No pattern in the last 10 candles.")
    if pst is not None:
        st.subheader(f"What happened after each daily pattern (1962–today, next {h} days)")
        s = pst[pst["h"] == h].drop(columns="h").copy()
        s["direction"] = s["direction"].fillna("")
        s["pattern"] = s["pattern"].str.replace("_", " ")
        for k in ("avg_ret", "avg_max_up", "avg_max_down"):
            s[k] = s[k] * 100
        s.columns = ["Pattern", "Signal", "Count", "Share up", "Avg return %", "Avg max up %", "Avg max down %"]
        st.dataframe(s.style.format({"Share up": "{:.1%}", "Avg return %": "{:+.2f}", "Avg max up %": "{:+.2f}",
                                     "Avg max down %": "{:+.2f}"}), hide_index=True, width="stretch")
        st.caption("Compare each row with **all days**: on the S&P 500, single candlestick patterns barely "
                   "change the direction. They say more about the *size* of the coming range: strong bearish "
                   "candles (three black crows, bearish marubozu) come before wider ranges, three white "
                   "soldiers before narrower ones.")

# --------------------------------------------------------------------------- method tab
with tab_how:
    if imp is not None:
        mm = st.radio("Importance from", ["xgboost", "catboost"], horizontal=True, format_func=MODEL_NAMES.get)
        g = imp[(imp["model"] == mm) & (imp["h"] == h)].groupby("feature")["importance"].mean().nlargest(15)[::-1]
        fig = go.Figure(go.Bar(x=g.values * 100, y=g.index, orientation="h", marker_color=C_MAX,
                               hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.update_layout(title=f"Top 15 features (share of importance, %), next {h} days, max + min averaged",
                          height=460, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    st.markdown(f"""
**Targets.** For each day *t*: `up = log(max high over t+1…t+{h} / close_t)` and
`down = log(min low over t+1…t+{h} / close_t)`. Max price = close × e^up, min price = close × e^down.

**Volatility scaling.** Models learn `target / σ`, with σ = Parkinson 21-day volatility × √(h/252),
so 1987, 2008 and 2017 are on one scale. Predictions are multiplied back by today's σ.

**Models.** XGBoost (`reg:quantileerror`) and CatBoost (`MultiQuantile`) predict the 10/50/90% quantiles;
the LSTM reads the last 30 days of features and outputs all quantiles of both targets with the pinball loss.
The ensemble averages the three.

**Candle resampling (multi-timeframe).** Daily candles are aggregated into weekly and monthly candles.
Those are mapped back (*upsampled*) to every day using only **already-closed** candles (last week's range,
body, pattern score, distance to last week's high / low) plus the still-forming week-to-date and month-to-date candle.

**Validation.** Walk-forward 2000–2021 (retrain every 2 years, 21-day gap), then a holdout from 2022
trained only on earlier data. Features pass a look-ahead check (rebuilt on truncated data = identical).
""")
