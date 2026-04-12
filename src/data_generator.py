import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHANNELS = {
    "Google Search": {
        "ctr_mean": 0.065, "ctr_std": 0.015,
        "lead_cvr_mean": 0.18, "lead_cvr_std": 0.04,
        "cust_cvr_mean": 0.22, "cust_cvr_std": 0.05,
        "cpc_mean": 2.8, "cpc_std": 0.6,
        "avg_revenue_per_customer": 420,
        "weight": 0.22,
    },
    "Meta Ads": {
        "ctr_mean": 0.032, "ctr_std": 0.010,
        "lead_cvr_mean": 0.11, "lead_cvr_std": 0.03,
        "cust_cvr_mean": 0.15, "cust_cvr_std": 0.04,
        "cpc_mean": 1.9, "cpc_std": 0.5,
        "avg_revenue_per_customer": 310,
        "weight": 0.25,
    },
    "LinkedIn": {
        "ctr_mean": 0.012, "ctr_std": 0.004,
        "lead_cvr_mean": 0.25, "lead_cvr_std": 0.06,
        "cust_cvr_mean": 0.28, "cust_cvr_std": 0.06,
        "cpc_mean": 7.5, "cpc_std": 1.8,
        "avg_revenue_per_customer": 1200,
        "weight": 0.12,
    },
    "Email": {
        "ctr_mean": 0.024, "ctr_std": 0.008,
        "lead_cvr_mean": 0.30, "lead_cvr_std": 0.07,
        "cust_cvr_mean": 0.35, "cust_cvr_std": 0.08,
        "cpc_mean": 0.4, "cpc_std": 0.1,
        "avg_revenue_per_customer": 380,
        "weight": 0.18,
    },
    "Affiliate": {
        "ctr_mean": 0.018, "ctr_std": 0.006,
        "lead_cvr_mean": 0.08, "lead_cvr_std": 0.03,
        "cust_cvr_mean": 0.10, "cust_cvr_std": 0.03,
        "cpc_mean": 1.2, "cpc_std": 0.4,
        "avg_revenue_per_customer": 270,
        "weight": 0.13,
    },
    "YouTube": {
        "ctr_mean": 0.009, "ctr_std": 0.003,
        "lead_cvr_mean": 0.07, "lead_cvr_std": 0.02,
        "cust_cvr_mean": 0.12, "cust_cvr_std": 0.03,
        "cpc_mean": 0.6, "cpc_std": 0.15,
        "avg_revenue_per_customer": 340,
        "weight": 0.10,
    },
}

CAMPAIGN_TEMPLATES = [
    ("Brand Awareness Q{q}", "brand_awareness"),
    ("Lead Gen — {product}", "lead_gen"),
    ("Retargeting — {segment}", "retargeting"),
    ("Seasonal Sale {season}", "promo"),
    ("Product Launch {product}", "launch"),
    ("Competitor Conquest", "conquest"),
    ("Lookalike Expansion", "prospecting"),
    ("Winback — Churned Users", "winback"),
]

PRODUCTS = ["Pro Plan", "Enterprise", "Starter", "Add-ons", "Annual Bundle"]
SEGMENTS = ["Trial Users", "High-Intent", "Dormant", "Engaged Free"]
REGIONS = ["North", "South", "East", "West", "Central"]
DEVICES = ["Mobile", "Desktop", "Tablet"]
AUDIENCES = ["18-24", "25-34", "35-44", "45-54", "55+"]


def seasonal_multiplier(date: datetime) -> float:
    """Q4 peaks, Q1/Q3 troughs."""
    month = date.month
    base = {1: 0.75, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.05,
            7: 0.90, 8: 0.85, 9: 1.00, 10: 1.10, 11: 1.35, 12: 1.40}
    return base[month]


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def generate_campaign_name(template_name: str, quarter: int) -> str:
    product = random.choice(PRODUCTS)
    segment = random.choice(SEGMENTS)
    season = ["Winter", "Spring", "Summer", "Fall"][quarter - 1]
    name = template_name.format(q=quarter, product=product, segment=segment, season=season)
    return name


def generate_data(start_date="2023-01-01", end_date="2024-12-31", num_campaigns=48) -> pd.DataFrame:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    campaigns = []
    campaign_id = 1000

    channel_names = list(CHANNELS.keys())
    channel_weights = [CHANNELS[c]["weight"] for c in channel_names]

    for i in range(num_campaigns):
        campaign_id += 1
        template_name, campaign_type = random.choice(CAMPAIGN_TEMPLATES)
        quarter = random.randint(1, 4)
        name = generate_campaign_name(template_name, quarter)
        channel = random.choices(channel_names, weights=channel_weights)[0]

        # Campaign duration: 14 to 90 days
        duration = random.randint(14, 90)
        camp_start = start + timedelta(days=random.randint(0, (end - start).days - duration))
        camp_end = min(camp_start + timedelta(days=duration), end)

        # Introduce intentional bad campaigns (~20%)
        is_bad = random.random() < 0.20
        # Introduce stellar campaigns (~15%)
        is_stellar = (not is_bad) and random.random() < 0.15

        ch = CHANNELS[channel]

        current = camp_start
        while current <= camp_end:
            s_mult = seasonal_multiplier(current)

            impressions_base = np.random.randint(8000, 150000)
            impressions = int(impressions_base * s_mult)

            ctr = clamp(np.random.normal(ch["ctr_mean"], ch["ctr_std"]), 0.001, 0.25)
            if is_bad:
                ctr *= 0.5
            if is_stellar:
                ctr *= 1.4

            clicks = max(1, int(impressions * ctr))

            cpc = max(0.05, np.random.normal(ch["cpc_mean"], ch["cpc_std"]))
            spend = round(clicks * cpc, 2)

            lead_cvr = clamp(np.random.normal(ch["lead_cvr_mean"], ch["lead_cvr_std"]), 0.01, 0.60)
            if is_bad:
                lead_cvr *= 0.55
            if is_stellar:
                lead_cvr *= 1.3

            leads = max(0, int(clicks * lead_cvr))

            cust_cvr = clamp(np.random.normal(ch["cust_cvr_mean"], ch["cust_cvr_std"]), 0.01, 0.70)
            if is_bad:
                cust_cvr *= 0.4
            if is_stellar:
                cust_cvr *= 1.35

            customers = max(0, int(leads * cust_cvr))

            rev_per_cust = np.random.normal(ch["avg_revenue_per_customer"], ch["avg_revenue_per_customer"] * 0.15)
            if is_bad:
                rev_per_cust *= 0.70
            revenue = round(max(0, customers * rev_per_cust), 2)

            campaigns.append({
                "date": current.strftime("%Y-%m-%d"),
                "campaign_id": f"C{campaign_id}",
                "campaign_name": name,
                "campaign_type": campaign_type,
                "channel": channel,
                "region": random.choice(REGIONS),
                "device": random.choice(DEVICES),
                "audience_segment": random.choice(AUDIENCES),
                "spend": spend,
                "impressions": impressions,
                "clicks": clicks,
                "leads": leads,
                "customers": customers,
                "revenue": revenue,
            })
            current += timedelta(days=1)

    df = pd.DataFrame(campaigns)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/raw/campaigns_raw.csv", index=False)
    print(f"Generated {len(df):,} rows across {df['campaign_id'].nunique()} campaigns")
    print(df.dtypes)
    print(df.head())