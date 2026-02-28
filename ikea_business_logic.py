"""
ikea_business_logic.py
======================
STEP 4 of 4 — Strategic Business Rules & Report Generation

Reads:   ikea_advanced_features.csv
Writes:  e:/Excel Datasets/Reports/ (targeted CSV files for different teams)

What it does:
  1. Translates ML clusters and features into plain-English business actions.
  2. Applies retail logic (e.g., BCG Matrix proxy, Promotion Alerts).
  3. Splits the master dataset into smaller, category-specific reports so 
     individual category managers (e.g., Head of Lighting) only see their products.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & LOAD
# ─────────────────────────────────────────────────────────────────────────────
FEAT_PATH   = Path("e:/Excel Datasets/ikea_advanced_features.csv")
REPORT_DIR  = Path("e:/Excel Datasets/Reports")

if not FEAT_PATH.exists():
    raise FileNotFoundError(f"{FEAT_PATH} not found. Run previous steps first.")

print(f"Loading enriched data from: {FEAT_PATH}")
df = pd.read_csv(FEAT_PATH, low_memory=False)

# ─────────────────────────────────────────────────────────────────────────────
# ADD BUSINESS LOGIC COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
print("Applying retail business logic...")

# 1. Promotional Margin Alert
# If a product is an organic hit (Top Seller or Community Favourite) but is 
# currently being promoted, we are likely throwing away margin.
df["promo_action"] = "Maintain Strategy"
mask_waste = (df["segment_name"].isin(["Community Favourites", "Top Sellers (Proven Hits)"])) & (df["promoted_share"] > 0)
df.loc[mask_waste, "promo_action"] = "FLAG: Cancel Promo (Margin Bleed)"

mask_need_push = (df["segment_name"] == "New Arrivals Pipeline") & (df["promoted_share"] == 0)
df.loc[mask_need_push, "promo_action"] = "FLAG: Add Introductory Promo"


# 2. Portfolio Strategy (Proxy for BCG Matrix)
# High Market Spread + High Engagement = Star / Cash Cow
# Low Market Spread + Low Engagement = Dog / Discontinue candidate
df["portfolio_role"] = "Core Assortment"

mask_star = (df["market_spread_share"] > 0.6) & (df["engagement_score"] > df["engagement_score"].quantile(0.75))
df.loc[mask_star, "portfolio_role"] = "Cash Cow (Protect Stock)"

mask_dog = (df["market_spread_share"] < 0.3) & (df["engagement_score"] < df["engagement_score"].quantile(0.25)) & (df["segment_name"] != "New Arrivals Pipeline")
df.loc[mask_dog, "portfolio_role"] = "Review for Discontinuation"

mask_expand = (df["market_spread_share"] < 0.2) & (df["engagement_score"] > df["engagement_score"].quantile(0.85))
df.loc[mask_expand, "portfolio_role"] = "Hidden Gem (Expand Globally)"


# 3. Product Size / Shipping Economics
# Flagging giant items that destroy e-commerce margins if not priced well
BIG_VOLUME = df["volume_est"].quantile(0.90) if "volume_est" in df.columns else float('inf')
df["logistics_flag"] = "Standard"
if BIG_VOLUME < float('inf'):
    df.loc[df["volume_est"] > BIG_VOLUME, "logistics_flag"] = "Bulky/Heavy (High Shipping Cost)"

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE TARGETED REPORTS
# ─────────────────────────────────────────────────────────────────────────────
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Define the columns a business manager actually wants to see (hide the ML math)
BUSINESS_COLS = [
    "product_id", "product_name", "main_category", "segment_name", 
    "portfolio_role", "promo_action", "logistics_flag",
    "market_spread", "avg_price_raw", "rating_imputed", 
    "engagement_score", "is_anomaly"
]
# Ensure we only select columns that exist
BUSINESS_COLS = [c for c in BUSINESS_COLS if c in df.columns]

business_df = df[BUSINESS_COLS].copy()

# 1. The Global Master Action Report
global_path = REPORT_DIR / "ikea_master_action_report.csv"
business_df.to_csv(global_path, index=False)
print(f"Saved Master Report: {global_path} ({len(business_df):,} products)")

# 2. The "Priority Action" list (Only the flagged items needing attention)
priority_df = business_df[
    business_df["promo_action"].str.contains("FLAG") | 
    business_df["portfolio_role"].isin(["Review for Discontinuation", "Hidden Gem (Expand Globally)"])
]
priority_path = REPORT_DIR / "ikea_priority_action_required.csv"
priority_df.to_csv(priority_path, index=False)
print(f"Saved Priority Action list: {priority_path} ({len(priority_df):,} items needing attention)")

print("\nBusiness logic applied successfully. Files are in e:/Excel Datasets/Reports/")
