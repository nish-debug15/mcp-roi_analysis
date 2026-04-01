# Marketing Campaign Performance + ROI Analysis

A practical, production-minded analytics project to evaluate marketing campaign effectiveness and make budget decisions with confidence.

## Why this project exists
Marketing teams often ask:
- Which campaigns are actually working?
- Where should we invest more budget?
- Which campaigns should we pause or stop?

This project answers those questions using a clean analytics workflow that combines performance metrics, unit economics, and decision rules.

## Business problem
Given campaign-level performance data across channels (Meta, Google, LinkedIn, Email, Affiliate, etc.), identify:
1. High-performing campaigns worth scaling.
2. Underperforming campaigns that should be optimized or stopped.
3. Channel-level opportunities based on CAC, ROI, and conversion efficiency.

## Core metrics
The analysis centers on metrics that stakeholders actually use:

- **Spend**: total ad cost.
- **Impressions**: number of times ads were shown.
- **Clicks**: number of ad clicks.
- **Leads**: captured prospects.
- **Customers**: converted paying users.
- **Revenue**: realized income from acquired customers.

Derived KPI formulas:

- **CTR** = `Clicks / Impressions`
- **CPC** = `Spend / Clicks`
- **Lead Conversion Rate** = `Leads / Clicks`
- **Customer Conversion Rate** = `Customers / Leads`
- **CAC (Customer Acquisition Cost)** = `Spend / Customers`
- **ROAS (Return on Ad Spend)** = `Revenue / Spend`
- **ROI** = `(Revenue - Spend) / Spend`

## Decision framework
Campaigns are classified with business rules (customizable):

- **Scale**: strong ROI, acceptable CAC, and stable conversion.
- **Optimize**: mixed performance; test creative, audience, bidding, or landing page.
- **Stop**: negative ROI and poor conversion over sufficient volume.

The framework avoids one-metric decisions by combining profitability and funnel quality.

## Target outputs
The final deliverables should include:

1. **Campaign scorecard**
   - campaign, channel, spend, revenue, CAC, ROI, ROAS, conversion rates
2. **Channel performance summary**
   - aggregated channel-wise economics
3. **Budget reallocation recommendations**
   - where to shift spend and expected impact
4. **Executive summary**
   - plain-language recommendations for leadership

## Data model (minimum schema)
Expected input columns:

- `date`
- `campaign_id`
- `campaign_name`
- `channel`
- `spend`
- `impressions`
- `clicks`
- `leads`
- `customers`
- `revenue`

Optional but useful:
- `region`
- `device`
- `audience_segment`
- `product_line`

## Suggested project structure
```text
mcp-roi_analysis/
├─ README.md
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
│  ├─ 01_data_quality.ipynb
│  ├─ 02_kpi_modeling.ipynb
│  └─ 03_recommendations.ipynb
├─ src/
│  ├─ io.py
│  ├─ metrics.py
│  ├─ scoring.py
│  └─ reporting.py
├─ outputs/
│  ├─ campaign_scorecard.csv
│  ├─ channel_summary.csv
│  └─ executive_summary.md
└─ requirements.txt
```

## Analysis workflow
1. **Ingest and validate data**
   - enforce schema, handle nulls, deduplicate rows, validate non-negative metrics.
2. **Compute KPIs**
   - derive CAC, ROI, ROAS, and conversion rates with divide-by-zero protection.
3. **Rank and segment campaigns**
   - performance tiers and rule-based labels.
4. **Aggregate by channel**
   - detect scalable channels and inefficient channels.
5. **Generate recommendations**
   - budget shift plan (increase / hold / reduce / stop).
6. **Publish outputs**
   - scorecard + concise executive summary.

## Quality and realism standards
To keep this project job-ready (not toy-level):

- Use explicit business assumptions and document them.
- Include data quality checks before KPI computation.
- Use robust rules for low-volume campaigns (avoid false positives).
- Separate metric calculation logic from reporting logic.
- Make recommendations traceable to numeric thresholds.

## Example recommendation logic
- If `ROI > 0.25` and `CAC <= target_cac` and `customers >= min_volume`: **Scale**
- If `ROI between 0 and 0.25`: **Optimize**
- If `ROI < 0` for consecutive periods and spend is meaningful: **Stop**

These are starter thresholds and should be tuned by business context.
