import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import pandas as pd

from data_generator import generate_data
from metrics import (
    compute_kpis, aggregate_campaign_kpis,
    aggregate_channel_kpis, score_campaigns, budget_reallocation,
    compute_monthly_trends
)
from ml_models import run_full_ml_pipeline
from reporting import (
    generate_campaign_scorecard,
    generate_channel_summary,
    generate_executive_summary
)


def run_pipeline(
    use_cached: bool = False,
    raw_path: str = "data/raw/campaigns_raw.csv",
    num_campaigns: int = 48,
):
    print("=" * 60)
    print("   MARKETING ROI ANALYSIS PIPELINE")
    print("=" * 60)

    print("\n[1/6] Generating / loading raw data...")
    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cached and raw_path.exists():
        df_raw = pd.read_csv(raw_path, parse_dates=["date"])
        print(f"  Loaded cached data: {len(df_raw):,} rows")
    else:
        df_raw = generate_data(num_campaigns=num_campaigns)
        df_raw.to_csv(raw_path, index=False)
        print(f"  Generated {len(df_raw):,} rows, {df_raw['campaign_id'].nunique()} campaigns")

    print("\n[2/6] Running data quality checks...")
    required_cols = ["date", "campaign_id", "campaign_name", "channel",
                     "spend", "impressions", "clicks", "leads", "customers", "revenue"]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    null_counts = df_raw[required_cols].isnull().sum()
    if null_counts.any():
        print(f"  Nulls found: {null_counts[null_counts > 0].to_dict()}")
        df_raw = df_raw.dropna(subset=required_cols)

    neg_mask = (df_raw[["spend", "impressions", "clicks", "leads", "customers", "revenue"]] < 0).any(axis=1)
    if neg_mask.any():
        print(f"  {neg_mask.sum()} rows with negative values — dropping")
        df_raw = df_raw[~neg_mask]

    dups = df_raw.duplicated(subset=["campaign_id", "date"]).sum()
    if dups > 0:
        print(f"  {dups} duplicate (campaign_id, date) rows — dropping")
        df_raw = df_raw.drop_duplicates(subset=["campaign_id", "date"])

    print(f"  Data quality OK — {len(df_raw):,} clean rows")

    # Save processed
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_raw.to_csv("data/processed/campaigns_clean.csv", index=False)

    print("\n[3/6] Computing KPIs...")
    campaign_df = aggregate_campaign_kpis(df_raw)
    campaign_df = score_campaigns(campaign_df)
    campaign_df = budget_reallocation(campaign_df)

    channel_df = aggregate_channel_kpis(df_raw)
    monthly_df = compute_monthly_trends(df_raw)

    print(f"  Campaigns: {len(campaign_df)} | Scale: {(campaign_df['decision']=='Scale').sum()} | "
          f"Optimize: {(campaign_df['decision']=='Optimize').sum()} | "
          f"Stop: {(campaign_df['decision']=='Stop').sum()}")

    print("\n[4/6] Running ML pipeline...")
    ml_results = run_full_ml_pipeline(campaign_df)
    campaign_df = ml_results["campaign_df"]

    print("\n[5/6] Exporting outputs...")
    Path("outputs").mkdir(exist_ok=True)

    scorecard = generate_campaign_scorecard(campaign_df, "outputs/campaign_scorecard.csv")
    channel_out = generate_channel_summary(channel_df, "outputs/channel_summary.csv")
    monthly_df.to_csv("outputs/monthly_trends.csv", index=False)
    ml_results["budget_plan"].to_csv("outputs/budget_reallocation.csv", index=False)

    exec_summary = generate_executive_summary(
        campaign_df, channel_df,
        ml_results["roi_metrics"],
        "outputs/executive_summary.md"
    )

    print("\n[6/6] Pipeline complete")
    print(f"\n{'─'*50}")
    total_spend = campaign_df["spend"].sum()
    total_revenue = campaign_df["revenue"].sum()
    print(f"  Portfolio spend:   ${total_spend:>12,.0f}")
    print(f"  Portfolio revenue: ${total_revenue:>12,.0f}")
    print(f"  Overall ROI:       {(total_revenue-total_spend)/total_spend*100:>10.1f}%")
    print(f"  Overall ROAS:      {total_revenue/total_spend:>11.2f}x")
    print(f"\n  ML model R²:       {ml_results['roi_metrics']['r2']:>11.4f}")
    print(f"  ML model MAE:      {ml_results['roi_metrics']['mae']:>11.4f}")
    print(f"{'─'*50}")
    print("\n  📁 Outputs written to ./outputs/")

    return {
        "raw": df_raw,
        "campaign_df": campaign_df,
        "channel_df": channel_df,
        "monthly_df": monthly_df,
        "ml_results": ml_results,
    }


if __name__ == "__main__":
    results = run_pipeline()