import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

CLUSTER_LABELS = {
    0: "Rising Stars",
    1: "Cash Cows",
    2: "Underperformers",
    3: "High-Volume Low-Margin",
}

FEATURE_COLS = ["ctr", "cpc", "lead_cvr", "customer_cvr", "cac", "roas", "roi",
                "spend", "impressions", "clicks", "customers", "perf_score"]


def cluster_campaigns(campaign_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    K-Means clustering on normalized KPI features.
    Returns df with cluster label and cluster name.
    """
    df = campaign_df.copy()
    available = [c for c in FEATURE_COLS if c in df.columns]

    X = df[available].fillna(0)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Map clusters to business labels by avg ROI
    cluster_roi = df.groupby("cluster")["roi"].mean().sort_values(ascending=False)
    roi_rank = {cid: rank for rank, cid in enumerate(cluster_roi.index)}

    label_map = {
        0: "Cash Cows",       
        1: "Rising Stars",    
        2: "Optimization Zone",
        3: "Underperformers", 
    }

    df["cluster_rank"] = df["cluster"].map(roi_rank)
    df["cluster_label"] = df["cluster_rank"].map(label_map)

    cluster_summary = df.groupby("cluster_label")[["roi", "roas", "cac", "spend", "revenue"]].mean().round(3)

    return df, cluster_summary, kmeans, scaler


def train_roi_predictor(campaign_df: pd.DataFrame):
    """
    Gradient Boosting model to predict ROI from early-funnel signals.
    Features: impressions, clicks, spend, ctr, cpc, lead_cvr
    Target: roi

    Returns trained model and evaluation metrics.
    """
    df = campaign_df.copy().dropna()

    feature_cols = ["spend", "impressions", "clicks", "ctr", "cpc", "lead_cvr", "customer_cvr"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["roi"]

    # Filter extreme ROI values
    mask = (y > -2) & (y < 20)
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "r2": round(r2_score(y_test, y_pred), 4),
        "feature_importance": dict(zip(
            feature_cols,
            model.named_steps["gbr"].feature_importances_.tolist()
        )),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return model, metrics, feature_cols


def detect_anomalies(campaign_df: pd.DataFrame, contamination: float = 0.10) -> pd.DataFrame:
    """
    Isolation Forest to flag statistically unusual campaigns.
    Anomalies may be either surprisingly good OR surprisingly bad.
    """
    df = campaign_df.copy()
    feature_cols = ["roi", "roas", "cac", "ctr", "cpc", "lead_cvr"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    df["anomaly_score"] = iso.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    df["anomaly_confidence"] = -iso.score_samples(X_scaled)  # higher = more anomalous
    df["is_anomaly"] = df["anomaly_score"] == -1

    median_roi = df["roi"].median()
    df["anomaly_type"] = np.where(
        ~df["is_anomaly"], "Normal",
        np.where(df["roi"] > median_roi, "Positive Outlier", "Negative Outlier")
    )

    return df


def optimize_budget(campaign_df: pd.DataFrame, total_budget: float) -> pd.DataFrame:
    """
    Greedy budget optimizer: allocate budget to maximize expected revenue.
    Uses ROI as a proxy for marginal return on incremental spend.
    
    Assumes diminishing returns (log model) for high-spend campaigns.
    """
    df = campaign_df.copy()
    df = df[df["decision"] != "Stop"].copy()

    df["rev_per_dollar"] = df["roas"].clip(0.1, 15)

    # Sort by efficiency descending
    df = df.sort_values("rev_per_dollar", ascending=False).reset_index(drop=True)

    remaining = total_budget
    allocations = []

    for _, row in df.iterrows():
        if remaining <= 0:
            allocations.append(0)
            continue
        max_alloc = min(row["spend"] * 3, total_budget * 0.30)
        alloc = min(max_alloc, remaining)
        allocations.append(round(alloc, 2))
        remaining -= alloc

    df["optimized_spend"] = allocations
    df["optimized_expected_revenue"] = df["optimized_spend"] * df["rev_per_dollar"]
    df["spend_delta"] = df["optimized_spend"] - df["spend"]

    return df[["campaign_id", "campaign_name", "channel", "decision",
               "spend", "optimized_spend", "spend_delta", "rev_per_dollar",
               "optimized_expected_revenue"]].sort_values("optimized_spend", ascending=False)


def run_full_ml_pipeline(campaign_df: pd.DataFrame):
    """
    Run all ML models and return a results bundle.
    """
    print("Running anomaly detection...")
    df_anomaly = detect_anomalies(campaign_df)

    print("Running campaign clustering...")
    df_clustered, cluster_summary, kmeans_model, scaler = cluster_campaigns(df_anomaly)

    print("Training ROI predictor...")
    roi_model, roi_metrics, feature_cols = train_roi_predictor(df_clustered)

    print("Running budget optimizer...")
    total_spend = campaign_df["spend"].sum()
    df_budget = optimize_budget(df_clustered, total_spend)

    print(f"ML pipeline complete. ROI model R²: {roi_metrics['r2']}")

    return {
        "campaign_df": df_clustered,
        "cluster_summary": cluster_summary,
        "roi_model": roi_model,
        "roi_metrics": roi_metrics,
        "roi_features": feature_cols,
        "budget_plan": df_budget,
    }