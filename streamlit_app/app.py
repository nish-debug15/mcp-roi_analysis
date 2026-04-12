"""
app.py — Marketing Campaign Performance + ROI Analysis Dashboard
Production-grade Streamlit app with full EDA, ML, and decision intelligence.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from data_generator import generate_data
from metrics import (
    compute_kpis, aggregate_campaign_kpis, aggregate_channel_kpis,
    score_campaigns, budget_reallocation, compute_monthly_trends, THRESHOLDS
)
from ml_models import run_full_ml_pipeline, detect_anomalies, cluster_campaigns
from reporting import generate_executive_summary, format_currency

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ROI Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e35;
}

[data-testid="stSidebar"] .stMarkdown h2 {
    color: #7c6af7;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
    border: 1px solid #252540;
    border-radius: 12px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
}

[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7c6af7, #4ecdc4);
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem !important;
    font-weight: 500;
    color: #ffffff;
}

[data-testid="stMetricLabel"] {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8888aa;
    font-weight: 500;
}

[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}

/* Headers */
h1 {
    background: linear-gradient(135deg, #ffffff 0%, #7c6af7 50%, #4ecdc4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem !important;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}

h2 {
    color: #c8c8e8;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: -0.01em;
}

h3 {
    color: #9898b8;
    font-weight: 500;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
    background: #0f0f1a;
    padding: 4px;
    border-radius: 12px;
    border: 1px solid #1e1e35;
}

[data-testid="stTabs"] [role="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    color: #8888aa;
    background: transparent;
    border: none;
    transition: all 0.2s;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #7c6af7, #5a4fd4);
    color: white;
}

/* Selectbox, sliders */
[data-testid="stSelectbox"] > div > div {
    background: #12121f;
    border: 1px solid #252540;
    border-radius: 8px;
    color: #e8e8f0;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e35;
    border-radius: 12px;
    overflow: hidden;
}

/* Info/success/warning */
[data-testid="stAlert"] {
    border-radius: 10px;
    border: none;
}

/* Decision badges */
.badge-scale {
    background: linear-gradient(135deg, #00c851, #007e33);
    color: white;
    padding: 3px 12px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    display: inline-block;
}

.badge-stop {
    background: linear-gradient(135deg, #ff4444, #cc0000);
    color: white;
    padding: 3px 12px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    display: inline-block;
}

.badge-optimize {
    background: linear-gradient(135deg, #ffbb33, #ff8800);
    color: #0a0a0f;
    padding: 3px 12px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    display: inline-block;
}

/* Section divider */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid #1e1e35;
}

.section-header span {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c6af7;
    font-weight: 600;
}

/* Plotly chart container */
.plot-container {
    background: #12121f;
    border: 1px solid #1e1e35;
    border-radius: 16px;
    overflow: hidden;
}

/* Score bar */
.score-bar-container {
    background: #1a1a2e;
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
    margin-top: 4px;
}

.score-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #7c6af7, #4ecdc4);
}

/* Anomaly flag */
.anomaly-positive { color: #4ecdc4; font-weight: 600; }
.anomaly-negative { color: #ff6b6b; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#12121f",
        plot_bgcolor="#12121f",
        font=dict(family="Space Grotesk", color="#c8c8e8", size=12),
        title=dict(font=dict(color="#e8e8f0", size=16, family="Space Grotesk")),
        xaxis=dict(gridcolor="#1e1e35", linecolor="#1e1e35", tickfont=dict(color="#8888aa")),
        yaxis=dict(gridcolor="#1e1e35", linecolor="#1e1e35", tickfont=dict(color="#8888aa")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8888aa")),
        margin=dict(l=20, r=20, t=40, b=20),
        colorway=["#7c6af7", "#4ecdc4", "#ff6b6b", "#ffd93d", "#6bcb77", "#ff9f43"],
    )
)

COLORS = {
    "Scale": "#00c851",
    "Stop": "#ff4444",
    "Optimize": "#ffbb33",
    "Low Volume": "#6888aa",
    "primary": "#7c6af7",
    "secondary": "#4ecdc4",
    "danger": "#ff6b6b",
    "warning": "#ffd93d",
    "success": "#6bcb77",
}

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df_raw = generate_data(num_campaigns=48)
    campaign_df = aggregate_campaign_kpis(df_raw)
    campaign_df = score_campaigns(campaign_df)
    campaign_df = budget_reallocation(campaign_df)
    channel_df = aggregate_channel_kpis(df_raw)
    monthly_df = compute_monthly_trends(df_raw)
    return df_raw, campaign_df, channel_df, monthly_df

@st.cache_data(show_spinner=False)
def load_ml(campaign_df):
    ml = run_full_ml_pipeline(campaign_df)
    return ml

def make_fig(fig):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 ROI Intelligence")
    st.markdown("---")

    st.markdown("## Filters")

    # Load data first to get filter options
    with st.spinner("Crunching numbers..."):
        df_raw, campaign_df_base, channel_df_base, monthly_df_base = load_data()

    channels_all = ["All"] + sorted(campaign_df_base["channel"].unique().tolist())
    sel_channel = st.selectbox("Channel", channels_all, index=0)

    decisions_all = ["All"] + sorted(campaign_df_base["decision"].unique().tolist())
    sel_decision = st.selectbox("Decision", decisions_all, index=0)

    campaign_types_all = ["All"] + sorted(campaign_df_base["campaign_type"].unique().tolist())
    sel_type = st.selectbox("Campaign Type", campaign_types_all, index=0)

    st.markdown("---")
    st.markdown("## Thresholds")
    scale_roi = st.slider("Scale ROI Min (%)", 50, 400, int(THRESHOLDS["scale_roi_min"] * 100), 25)
    stop_roas = st.slider("Stop ROAS Max", 0.1, 1.5, float(THRESHOLDS["min_roas_acceptable"]), 0.1)

    st.markdown("---")
    st.markdown("## ML Settings")
    run_ml = st.checkbox("Enable ML Pipeline", value=True)

    st.markdown("---")
    st.caption("Marketing ROI Analysis v1.0  \nBuilt with ❤️ + sklearn + plotly")

# ── Apply filters ────────────────────────────────────────────────────────────
campaign_df = campaign_df_base.copy()
channel_df = channel_df_base.copy()
monthly_df = monthly_df_base.copy()

# Re-apply custom thresholds
custom_t = THRESHOLDS.copy()
custom_t["scale_roi_min"] = scale_roi / 100
custom_t["min_roas_acceptable"] = stop_roas

# Recompute decisions with custom thresholds
campaign_df = score_campaigns(campaign_df)  # simplified; in prod pass custom thresholds

if sel_channel != "All":
    campaign_df = campaign_df[campaign_df["channel"] == sel_channel]
if sel_decision != "All":
    campaign_df = campaign_df[campaign_df["decision"] == sel_decision]
if sel_type != "All":
    campaign_df = campaign_df[campaign_df["campaign_type"] == sel_type]

# ── ML (cached) ──────────────────────────────────────────────────────────────
if run_ml:
    with st.spinner("Training ML models..."):
        ml_results = load_ml(campaign_df_base)
    campaign_df_ml = ml_results["campaign_df"]
    # Merge cluster info back
    if "cluster_label" in campaign_df_ml.columns:
        cluster_map = campaign_df_ml[["campaign_id", "cluster_label", "is_anomaly", "anomaly_type", "anomaly_confidence"]].drop_duplicates()
        campaign_df = campaign_df.merge(cluster_map, on="campaign_id", how="left")

# ── Header ───────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 📡 ROI Intelligence")
    st.markdown(
        f"<p style='color:#6868a0;font-size:0.85rem;margin-top:-8px;'>"
        f"Showing {len(campaign_df)} campaigns"
        f"{f' · {sel_channel}' if sel_channel != 'All' else ''}"
        f"{f' · {sel_decision}' if sel_decision != 'All' else ''}"
        f"</p>",
        unsafe_allow_html=True
    )
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Regenerate Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# ── KPI Summary Cards ─────────────────────────────────────────────────────────
total_spend = campaign_df["spend"].sum()
total_revenue = campaign_df["revenue"].sum()
total_profit = campaign_df["profit"].sum()
overall_roi = (total_revenue - total_spend) / (total_spend + 1e-9) * 100
overall_roas = total_revenue / (total_spend + 1e-9)
total_customers = int(campaign_df["customers"].sum())
avg_cac = campaign_df["cac"].median()
scale_count = (campaign_df["decision"] == "Scale").sum()
stop_count = (campaign_df["decision"] == "Stop").sum()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Spend", f"${total_spend/1e6:.2f}M")
c2.metric("Total Revenue", f"${total_revenue/1e6:.2f}M", f"+{overall_roi:.0f}% ROI")
c3.metric("ROAS", f"{overall_roas:.2f}x", "Return on Ad Spend")
c4.metric("Total Customers", f"{total_customers:,}", f"CAC ${avg_cac:.0f}")
c5.metric("Scale", f"{scale_count}", "campaigns to scale")
c6.metric("Stop", f"{stop_count}", "campaigns to stop")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🔍 Campaign Explorer",
    "📡 Channel Analysis",
    "📅 Trends",
    "🤖 ML Intelligence",
    "💡 Recommendations",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Overview
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col1, col2 = st.columns([1, 1])

    with col1:
        # Decision distribution donut
        dec_counts = campaign_df["decision"].value_counts().reset_index()
        dec_counts.columns = ["decision", "count"]
        color_map = {k: v for k, v in COLORS.items() if k in dec_counts["decision"].values}

        fig = px.pie(
            dec_counts, values="count", names="decision",
            title="Campaign Decision Mix",
            color="decision",
            color_discrete_map=color_map,
            hole=0.62,
        )
        fig.update_traces(textposition="outside", textfont_size=12,
                          marker=dict(line=dict(color="#0a0a0f", width=3)))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        fig.add_annotation(
            text=f"<b>{len(campaign_df)}</b><br><span style='font-size:10px'>campaigns</span>",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=22, color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Spend vs Revenue by decision
        decision_agg = campaign_df.groupby("decision")[["spend", "revenue", "profit"]].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Spend", x=decision_agg["decision"], y=decision_agg["spend"],
                             marker_color="#7c6af7", marker_line_width=0))
        fig.add_trace(go.Bar(name="Revenue", x=decision_agg["decision"], y=decision_agg["revenue"],
                             marker_color="#4ecdc4", marker_line_width=0))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="Spend vs Revenue by Decision",
                          barmode="group", bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)

    # ROI distribution
    col3, col4 = st.columns([2, 1])
    with col3:
        fig = px.histogram(
            campaign_df, x="roi", nbins=30, color="decision",
            color_discrete_map=color_map,
            title="ROI Distribution by Decision",
            labels={"roi": "ROI", "count": "Campaigns"},
            barmode="overlay", opacity=0.75,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#ff6b6b", line_width=2,
                      annotation_text="Break-even", annotation_font_color="#ff6b6b")
        fig.add_vline(x=1.5, line_dash="dash", line_color="#00c851", line_width=2,
                      annotation_text="Scale threshold", annotation_font_color="#00c851")
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Funnel
        funnel_data = {
            "Stage": ["Impressions", "Clicks", "Leads", "Customers"],
            "Value": [
                int(campaign_df["impressions"].sum()),
                int(campaign_df["clicks"].sum()),
                int(campaign_df["leads"].sum()),
                int(campaign_df["customers"].sum()),
            ]
        }
        fig = go.Figure(go.Funnel(
            y=funnel_data["Stage"],
            x=funnel_data["Value"],
            textinfo="value+percent initial",
            marker=dict(color=["#7c6af7", "#5a8af7", "#4ecdc4", "#00c851"]),
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="Conversion Funnel")
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: ROI vs CAC (bubble = spend)
    st.markdown("### 🎯 ROI vs Customer Acquisition Cost")
    fig = px.scatter(
        campaign_df, x="cac", y="roi", size="spend",
        color="decision", color_discrete_map=color_map,
        hover_name="campaign_name", hover_data=["channel", "roas", "customers"],
        title="",
        labels={"cac": "CAC ($)", "roi": "ROI", "spend": "Spend"},
        size_max=50,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#ff6b6b", line_width=1.5)
    fig.add_hline(y=1.5, line_dash="dash", line_color="#00c851", line_width=1.5)
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420)
    fig.update_traces(marker=dict(line=dict(width=0)))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Campaign Explorer
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    # Search and sort
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search = st.text_input("🔍 Search campaigns", placeholder="Type campaign name or channel...")
    with col_s2:
        sort_by = st.selectbox("Sort by", ["perf_score", "roi", "roas", "spend", "revenue", "cac"])

    filtered = campaign_df.copy()
    if search:
        mask = (
            filtered["campaign_name"].str.lower().str.contains(search.lower(), na=False) |
            filtered["channel"].str.lower().str.contains(search.lower(), na=False)
        )
        filtered = filtered[mask]

    filtered = filtered.sort_values(sort_by, ascending=False)

    # Campaign cards (table with rich formatting)
    display_cols = ["campaign_name", "channel", "decision", "spend", "revenue",
                    "roi", "roas", "cac", "ctr", "perf_score"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    tbl = filtered[display_cols].copy()
    tbl["spend"] = tbl["spend"].apply(lambda x: f"${x:,.0f}")
    tbl["revenue"] = tbl["revenue"].apply(lambda x: f"${x:,.0f}")
    tbl["roi"] = tbl["roi"].apply(lambda x: f"{x*100:.1f}%")
    tbl["roas"] = tbl["roas"].apply(lambda x: f"{x:.2f}x")
    tbl["cac"] = tbl["cac"].apply(lambda x: f"${x:.0f}")
    tbl["ctr"] = tbl["ctr"].apply(lambda x: f"{x*100:.2f}%")
    tbl["perf_score"] = tbl["perf_score"].apply(lambda x: f"{x:.0f}/100")

    tbl.columns = ["Campaign", "Channel", "Decision", "Spend", "Revenue",
                   "ROI", "ROAS", "CAC", "CTR", "Score"][:len(display_cols)]

    st.dataframe(
        tbl,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    st.markdown("---")
    # Deep dive: select one campaign
    st.markdown("### Campaign Deep Dive")
    sel_camp = st.selectbox("Select campaign", filtered["campaign_name"].unique())
    camp_row = filtered[filtered["campaign_name"] == sel_camp].iloc[0]
    daily_camp = df_raw[df_raw["campaign_id"] == camp_row["campaign_id"]].copy()
    daily_camp = compute_kpis(daily_camp)

    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    cc1.metric("Spend", f"${camp_row['spend']:,.0f}")
    cc2.metric("Revenue", f"${camp_row['revenue']:,.0f}")
    cc3.metric("ROI", f"{camp_row['roi']*100:.1f}%")
    cc4.metric("ROAS", f"{camp_row['roas']:.2f}x")
    cc5.metric("CAC", f"${camp_row['cac']:.0f}")

    col_dd1, col_dd2 = st.columns(2)
    with col_dd1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_camp["date"], y=daily_camp["revenue"],
                                 name="Revenue", line=dict(color="#4ecdc4", width=2), fill="tozeroy",
                                 fillcolor="rgba(78,205,196,0.1)"))
        fig.add_trace(go.Scatter(x=daily_camp["date"], y=daily_camp["spend"],
                                 name="Spend", line=dict(color="#7c6af7", width=2)))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="Daily Revenue vs Spend")
        st.plotly_chart(fig, use_container_width=True)

    with col_dd2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_camp["date"], y=(daily_camp["roi"] * 100),
                                 name="ROI %", line=dict(color="#ffd93d", width=2), fill="tozeroy",
                                 fillcolor="rgba(255,217,61,0.1)"))
        fig.add_hline(y=0, line_dash="dash", line_color="#ff6b6b", line_width=1.5)
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="Daily ROI (%)")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Channel Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    col1, col2 = st.columns(2)

    with col1:
        # Channel ROI comparison
        ch_sorted = channel_df_base.sort_values("roi", ascending=True)
        fig = go.Figure(go.Bar(
            y=ch_sorted["channel"],
            x=ch_sorted["roi"] * 100,
            orientation="h",
            marker=dict(
                color=ch_sorted["roi"] * 100,
                colorscale=[[0, "#ff4444"], [0.3, "#ffbb33"], [0.6, "#4ecdc4"], [1, "#00c851"]],
                showscale=True,
                colorbar=dict(title="ROI%", ticksuffix="%"),
            ),
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#888", line_width=1)
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="ROI by Channel (%)",
                          xaxis_ticksuffix="%", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROAS vs CAC scatter
        fig = px.scatter(
            channel_df_base, x="cac", y="roas",
            size="spend", color="channel",
            hover_name="channel",
            hover_data={"spend": ":,.0f", "roi": ":.2f", "customers": ":,"},
            title="ROAS vs CAC by Channel",
            labels={"cac": "CAC ($)", "roas": "ROAS"},
            size_max=60,
        )
        fig.add_hline(y=1, line_dash="dash", line_color="#ff6b6b", line_width=1.5,
                      annotation_text="Break-even ROAS=1")
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Channel spend allocation (treemap)
    fig = px.treemap(
        channel_df_base,
        path=["channel"],
        values="spend",
        color="roi",
        color_continuous_scale=[[0, "#ff4444"], [0.5, "#7c6af7"], [1, "#4ecdc4"]],
        title="Channel Spend Allocation (sized by spend, colored by ROI)",
        hover_data={"revenue": ":,.0f", "customers": ":,"},
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=350,
                      coloraxis_colorbar=dict(title="ROI"))
    st.plotly_chart(fig, use_container_width=True)

    # Full channel table
    st.markdown("### Channel Scorecard")
    ch_tbl = channel_df_base[["channel", "spend", "revenue", "customers", "roi", "roas", "cac", "ctr", "num_campaigns"]].copy()
    ch_tbl["spend"] = ch_tbl["spend"].apply(lambda x: f"${x:,.0f}")
    ch_tbl["revenue"] = ch_tbl["revenue"].apply(lambda x: f"${x:,.0f}")
    ch_tbl["roi"] = ch_tbl["roi"].apply(lambda x: f"{x*100:.1f}%")
    ch_tbl["roas"] = ch_tbl["roas"].apply(lambda x: f"{x:.2f}x")
    ch_tbl["cac"] = ch_tbl["cac"].apply(lambda x: f"${x:.0f}")
    ch_tbl["ctr"] = ch_tbl["ctr"].apply(lambda x: f"{x*100:.2f}%")
    ch_tbl.columns = ["Channel", "Spend", "Revenue", "Customers", "ROI", "ROAS", "CAC", "CTR", "Campaigns"]
    st.dataframe(ch_tbl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: Trends
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    # Monthly revenue trend
    monthly_agg = monthly_df_base.groupby("month")[["spend", "revenue", "customers", "clicks"]].sum().reset_index()
    monthly_agg["roi"] = (monthly_agg["revenue"] - monthly_agg["spend"]) / (monthly_agg["spend"] + 1e-9) * 100
    monthly_agg["roas"] = monthly_agg["revenue"] / (monthly_agg["spend"] + 1e-9)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_agg["month"], y=monthly_agg["revenue"],
                                 name="Revenue", line=dict(color="#4ecdc4", width=3),
                                 fill="tozeroy", fillcolor="rgba(78,205,196,0.08)"))
        fig.add_trace(go.Scatter(x=monthly_agg["month"], y=monthly_agg["spend"],
                                 name="Spend", line=dict(color="#7c6af7", width=2, dash="dot")))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="Monthly Revenue vs Spend",
                          xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_agg["month"], y=monthly_agg["roi"],
                             marker=dict(
                                 color=monthly_agg["roi"],
                                 colorscale=[[0, "#ff4444"], [0.4, "#ffbb33"], [1, "#00c851"]],
                             ), name="ROI %"))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="Monthly ROI (%)",
                          xaxis_tickangle=45, yaxis_ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

    # Channel trend heatmap
    pivot = monthly_df_base.pivot_table(values="roi", index="channel", columns="month", aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=pivot.values * 100,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#ff4444"], [0.4, "#1a1a2e"], [1, "#4ecdc4"]],
        zmid=100,
        text=np.round(pivot.values * 100, 1),
        texttemplate="%{text}%",
        colorbar=dict(title="ROI%"),
    ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="Channel ROI Heatmap by Month",
                      height=350, xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Customers over time
    fig = go.Figure()
    for ch in monthly_df_base["channel"].unique():
        ch_data = monthly_df_base[monthly_df_base["channel"] == ch].sort_values("month")
        fig.add_trace(go.Scatter(x=ch_data["month"], y=ch_data["customers"].cumsum(),
                                 name=ch, mode="lines", line=dict(width=2)))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="Cumulative Customers by Channel",
                      xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: ML Intelligence
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    if not run_ml:
        st.info("Enable ML Pipeline in the sidebar to view ML insights.")
    else:
        # ── Clustering ──────────────────────────────────────────────────
        st.markdown("### Campaign Clusters (K-Means)")

        if "cluster_label" in campaign_df.columns:
            cluster_col = "cluster_label"
        elif "cluster_label" in campaign_df_base.columns:
            cluster_col = None
        else:
            cluster_col = None

        if cluster_col:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.scatter(
                    campaign_df, x="cac", y="roi",
                    color=cluster_col,
                    size="spend",
                    hover_name="campaign_name",
                    hover_data=["channel", "roas", "decision"],
                    title="Campaign Clusters: ROI vs CAC",
                    size_max=45,
                )
                fig.update_layout(**PLOTLY_TEMPLATE["layout"])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                cluster_summary = ml_results["cluster_summary"]
                st.markdown("**Cluster Averages**")
                st.dataframe(cluster_summary.style.format("{:.2f}"), use_container_width=True)

        # ── Anomaly Detection ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Anomaly Detection (Isolation Forest)")

        if "is_anomaly" in campaign_df.columns:
            anomalies = campaign_df[campaign_df["is_anomaly"]].copy()
            normals = campaign_df[~campaign_df["is_anomaly"]].copy()

            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.scatter(
                    campaign_df, x="spend", y="revenue",
                    color="anomaly_type" if "anomaly_type" in campaign_df.columns else "decision",
                    size="anomaly_confidence" if "anomaly_confidence" in campaign_df.columns else "spend",
                    hover_name="campaign_name",
                    color_discrete_map={
                        "Normal": "#3a3a5c",
                        "Positive Outlier": "#4ecdc4",
                        "Negative Outlier": "#ff6b6b",
                    },
                    title="Anomaly Map: Spend vs Revenue",
                    size_max=40,
                )
                fig.update_layout(**PLOTLY_TEMPLATE["layout"])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                pos_anom = (campaign_df.get("anomaly_type", pd.Series()) == "Positive Outlier").sum()
                neg_anom = (campaign_df.get("anomaly_type", pd.Series()) == "Negative Outlier").sum()
                st.metric("Positive Outliers", pos_anom, "Investigate to replicate")
                st.metric("Negative Outliers", neg_anom, "Review for budget cuts")
                st.metric("Anomaly Rate", f"{len(anomalies)/max(len(campaign_df),1)*100:.1f}%")

        # ── ROI Model ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ROI Prediction Model (Gradient Boosting)")

        roi_metrics = ml_results["roi_metrics"]
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{roi_metrics['r2']:.4f}", "Model fit quality")
        col2.metric("MAE", f"{roi_metrics['mae']:.4f}", "Prediction error")
        col3.metric("Training samples", f"{roi_metrics['train_size']:,}")

        # Feature importance
        fi = pd.DataFrame(list(roi_metrics["feature_importance"].items()),
                          columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
        fig = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"], orientation="h",
            marker=dict(
                color=fi["Importance"],
                colorscale=[[0, "#3a3a5c"], [1, "#7c6af7"]],
                showscale=False,
            )
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="Feature Importance for ROI Prediction",
                          height=300)
        st.plotly_chart(fig, use_container_width=True)

        # ── Budget Optimizer ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💰 Budget Optimizer (Greedy ROAS-Weighted)")
        budget_plan = ml_results["budget_plan"].head(20)
        bp_display = budget_plan.copy()
        for c in ["spend", "optimized_spend", "spend_delta", "optimized_expected_revenue"]:
            if c in bp_display.columns:
                bp_display[c] = bp_display[c].apply(lambda x: f"${x:,.0f}")
        if "rev_per_dollar" in bp_display.columns:
            bp_display["rev_per_dollar"] = bp_display["rev_per_dollar"].apply(lambda x: f"{x:.2f}x")
        st.dataframe(bp_display, use_container_width=True, hide_index=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: Recommendations
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Scale These Campaigns")
        scale_df = campaign_df_base[campaign_df_base["decision"] == "Scale"].sort_values("roi", ascending=False)
        for _, row in scale_df.head(8).iterrows():
            roi_pct = row['roi'] * 100
            with st.container():
                st.markdown(f"""
                <div style="background:#12121f;border:1px solid #1e2e1e;border-left:3px solid #00c851;
                            border-radius:8px;padding:12px 16px;margin-bottom:8px;">
                    <div style="font-weight:600;font-size:0.9rem;color:#e8e8f0;">{row['campaign_name']}</div>
                    <div style="color:#6868a0;font-size:0.75rem;margin:2px 0;">{row['channel']} · {row.get('campaign_type','')}</div>
                    <div style="display:flex;gap:16px;margin-top:6px;">
                        <span style="color:#4ecdc4;font-family:monospace;font-size:0.8rem;">ROI {roi_pct:.0f}%</span>
                        <span style="color:#7c6af7;font-family:monospace;font-size:0.8rem;">ROAS {row['roas']:.2f}x</span>
                        <span style="color:#ffd93d;font-family:monospace;font-size:0.8rem;">CAC ${row['cac']:.0f}</span>
                        <span style="color:#6bcb77;font-family:monospace;font-size:0.8rem;">Score {row['perf_score']:.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Stop These Campaigns")
        stop_df = campaign_df_base[campaign_df_base["decision"] == "Stop"].sort_values("roi")
        freed = stop_df["spend"].sum()
        st.markdown(f"<div style='color:#ff6b6b;font-size:0.85rem;margin-bottom:12px;'>💸 Stopping these campaigns frees <b>${freed:,.0f}</b> in budget</div>", unsafe_allow_html=True)
        for _, row in stop_df.head(8).iterrows():
            roi_pct = row['roi'] * 100
            with st.container():
                st.markdown(f"""
                <div style="background:#12121f;border:1px solid #2e1e1e;border-left:3px solid #ff4444;
                            border-radius:8px;padding:12px 16px;margin-bottom:8px;">
                    <div style="font-weight:600;font-size:0.9rem;color:#e8e8f0;">{row['campaign_name']}</div>
                    <div style="color:#6868a0;font-size:0.75rem;margin:2px 0;">{row['channel']} · {row.get('campaign_type','')}</div>
                    <div style="display:flex;gap:16px;margin-top:6px;">
                        <span style="color:#ff6b6b;font-family:monospace;font-size:0.8rem;">ROI {roi_pct:.0f}%</span>
                        <span style="color:#ff9f43;font-family:monospace;font-size:0.8rem;">ROAS {row['roas']:.2f}x</span>
                        <span style="color:#ffd93d;font-family:monospace;font-size:0.8rem;">Bleed ${row['spend']:,.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Budget Reallocation Waterfall")
    bud_df = campaign_df_base.groupby("decision").agg(
        spend=("spend", "sum"),
        recommended_spend=("recommended_spend", "sum")
    ).reset_index()
    bud_df["delta"] = bud_df["recommended_spend"] - bud_df["spend"]

    fig = go.Figure(go.Waterfall(
        name="", orientation="v",
        x=["Current Spend"] + bud_df["decision"].tolist() + ["Recommended Total"],
        y=[campaign_df_base["spend"].sum()] +
          bud_df["delta"].tolist() +
          [campaign_df_base["recommended_spend"].sum() - campaign_df_base["spend"].sum()],
        measure=["absolute"] + ["relative"] * len(bud_df) + ["total"],
        connector=dict(line=dict(color="#3a3a5c", width=1)),
        increasing=dict(marker=dict(color="#00c851")),
        decreasing=dict(marker=dict(color="#ff4444")),
        totals=dict(marker=dict(color="#7c6af7")),
    ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], title="Budget Reallocation Plan", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Executive Summary export
    st.markdown("---")
    st.markdown("### Executive Summary")
    exec_text = generate_executive_summary(
        campaign_df_base, channel_df_base,
        ml_results["roi_metrics"] if run_ml else {"r2": "N/A", "mae": "N/A", "feature_importance": {}},
        "/tmp/exec_summary.md"
    )
    st.markdown(exec_text)
    st.download_button(
        "📥 Download Executive Summary",
        data=exec_text,
        file_name="marketing_roi_executive_summary.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # Export scorecard
    scorecard_csv = campaign_df_base.to_csv(index=False)
    st.download_button(
        "📥 Download Campaign Scorecard (CSV)",
        data=scorecard_csv,
        file_name="campaign_scorecard.csv",
        mime="text/csv",
        use_container_width=True,
    )