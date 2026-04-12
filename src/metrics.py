import pandas as pd
import numpy as np
from typing import Optional

THRESHOLDS = {
    "min_clicks_for_analysis": 100,
    "min_customers_for_analysis": 5,     
    "scale_roi_min": 1.50,               
    "scale_cac_max_ratio": 1.2,          
    "stop_roi_max": 0.0,                 
    "optimize_roi_min": 0.0,             
    "min_roas_acceptable": 1.0,          
    "strong_ctr_threshold": 0.05,        
    "poor_ctr_threshold": 0.01,          
}

EPS = 1e-9  


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived KPIs from raw campaign columns.
    Input must contain: spend, impressions, clicks, leads, customers, revenue.
    Returns df with new KPI columns appended.
    """
    df = df.copy()

    df["ctr"] = df["clicks"] / (df["impressions"] + EPS)
    df["cpc"] = df["spend"] / (df["clicks"] + EPS)
    df["lead_cvr"] = df["leads"] / (df["clicks"] + EPS)
    df["customer_cvr"] = df["customers"] / (df["leads"] + EPS)
    df["overall_cvr"] = df["customers"] / (df["clicks"] + EPS)

    df["cac"] = df["spend"] / (df["customers"] + EPS)
    df["roas"] = df["revenue"] / (df["spend"] + EPS)
    df["roi"] = (df["revenue"] - df["spend"]) / (df["spend"] + EPS)
    df["revenue_per_lead"] = df["revenue"] / (df["leads"] + EPS)
    df["profit"] = df["revenue"] - df["spend"]

    for col in ["cac", "cpc", "ctr", "roas", "roi"]:
        upper = df[col].quantile(0.99)
        lower = df[col].quantile(0.01)
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def aggregate_campaign_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily rows to campaign-level summary.
    Uses sum for volume metrics, then recomputes rates from aggregates (correct approach).
    """
    agg = df.groupby(
        ["campaign_id", "campaign_name", "campaign_type", "channel"],
        as_index=False
    ).agg(
        spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        leads=("leads", "sum"),
        customers=("customers", "sum"),
        revenue=("revenue", "sum"),
        days_active=("date", "nunique"),
        start_date=("date", "min"),
        end_date=("date", "max"),
    )

    agg = compute_kpis(agg)
    return agg


def aggregate_channel_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Channel-level rollup with median CAC and benchmark comparison."""
    agg = df.groupby("channel", as_index=False).agg(
        spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        leads=("leads", "sum"),
        customers=("customers", "sum"),
        revenue=("revenue", "sum"),
        num_campaigns=("campaign_id", "nunique"),
    )
    agg = compute_kpis(agg)

    campaign_agg = aggregate_campaign_kpis(df)
    median_cac = campaign_agg.groupby("channel")["cac"].median().reset_index()
    median_cac.columns = ["channel", "median_cac"]
    agg = agg.merge(median_cac, on="channel", how="left")

    return agg


def score_campaigns(campaign_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply decision rules to classify campaigns:
    - Scale: ROI > threshold AND adequate volume
    - Stop:  ROI ≤ 0 AND adequate volume
    - Optimize: everything in between
    - Low Volume: insufficient data for decision

    Returns campaign_df with 'decision', 'confidence', and 'score' columns.
    """
    df = campaign_df.copy()
    t = THRESHOLDS

    low_volume = (
        (df["clicks"] < t["min_clicks_for_analysis"]) |
        (df["customers"] < t["min_customers_for_analysis"])
    )

    scale = (
        ~low_volume &
        (df["roi"] >= t["scale_roi_min"]) &
        (df["roas"] >= t["min_roas_acceptable"])
    )

    stop = (
        ~low_volume &
        (df["roi"] <= t["stop_roi_max"]) &
        (df["roas"] < t["min_roas_acceptable"])
    )

    optimize = ~low_volume & ~scale & ~stop

    df["decision"] = np.select(
        [low_volume, scale, stop, optimize],
        ["Low Volume", "Scale", "Stop", "Optimize"],
        default="Optimize"
    )

    roi_norm = df["roi"].clip(-1, 5) / 5
    roas_norm = (df["roas"] - 1).clip(0, 10) / 10
    cvr_norm = df["overall_cvr"].clip(0, 0.1) / 0.1
    ctr_norm = df["ctr"].clip(0, 0.15) / 0.15

    df["perf_score"] = (
        0.40 * roi_norm +
        0.30 * roas_norm +
        0.20 * cvr_norm +
        0.10 * ctr_norm
    ).clip(0, 1) * 100

    return df


def budget_reallocation(campaign_df: pd.DataFrame, total_budget: Optional[float] = None) -> pd.DataFrame:
    """
    Generate budget reallocation recommendations.
    - Scale campaigns get proportionally more budget based on ROI rank.
    - Stop campaigns get zero new budget.
    - Optimize campaigns maintain current spend.
    """
    df = campaign_df.copy()

    if total_budget is None:
        total_budget = df["spend"].sum()

    scale_mask = df["decision"] == "Scale"
    stop_mask = df["decision"] == "Stop"
    optimize_mask = df["decision"] == "Optimize"

    freed = df.loc[stop_mask, "spend"].sum()
    available_extra = freed * 0.80  

    scale_roi = df.loc[scale_mask, "roi"]
    scale_total_roi = scale_roi.sum()

    df["recommended_spend"] = df["spend"]  
    df.loc[stop_mask, "recommended_spend"] = 0

    if scale_total_roi > 0 and available_extra > 0:
        scale_idx = df.index[scale_mask]
        df.loc[scale_idx, "recommended_spend"] = (
            df.loc[scale_idx, "spend"] +
            (df.loc[scale_idx, "roi"] / scale_total_roi) * available_extra
        )

    df["spend_change"] = df["recommended_spend"] - df["spend"]
    df["spend_change_pct"] = df["spend_change"] / (df["spend"] + EPS) * 100

    return df


def compute_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregated performance for trend analysis."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    monthly = df.groupby(["month", "channel"], as_index=False).agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum"),
        clicks=("clicks", "sum"),
        customers=("customers", "sum"),
        impressions=("impressions", "sum"),
        leads=("leads", "sum"),
    )
    monthly = compute_kpis(monthly)
    monthly["month"] = monthly["month"].astype(str)
    return monthly