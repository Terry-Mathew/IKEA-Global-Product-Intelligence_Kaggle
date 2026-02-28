"""
ikea_cleaner.py
===============
STEP 1 of 3 — Data Cleaning & Transformation

Reads:   IKEA_product_catalog.csv  (raw scraped data)
Writes:  ikea_clean.csv            (cleaned, ready for feature engineering)

Run this FIRST before ikea_advanced_ds.py or ikea_analysis.py.

What it does:
  1. Normalises sentinel strings ("none", "nan") to real NaN
  2. Converts price / rating / review_count to numeric
  3. Removes impossible prices (<=0 or above 99th pct within currency)
  4. Corrects out-of-range ratings (IKEA scale is 1–5)
  5. Removes duplicate listing rows (same unique_id scraped twice)
  6. Flags actually-missing vs. zero-review products separately
  7. Applies log1p to right-skewed columns before saving
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
RAW_PATH  = Path("e:/Excel Datasets/IKEA_product_catalog.csv")
OUT_PATH  = Path("e:/Excel Datasets/ikea_clean.csv")

# Columns to log-transform (right-skewed; log1p-safe because all >= 0)
# These are raw columns to fix BEFORE downstream scripts use them.
LOG_TRANSFORM_COLS = ["price", "product_rating_count"]

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
print(f"Loading: {RAW_PATH}")
df = pd.read_csv(RAW_PATH, low_memory=False)
n_raw = len(df)
print(f"  Raw rows: {n_raw:,}")

# ─────────────────────────────────────────────────────────────
# STEP 1 — Normalise sentinel strings → NaN globally
# The CSV uses literal "none" to encode missing in many columns.
# Replacing at string level before any to_numeric() call ensures
# coerce doesn't silently swallow legitimate errors.
# ─────────────────────────────────────────────────────────────
SENTINEL_STRINGS = {"none", "nan", "n/a", "null", "", "#n/a"}
for col in df.select_dtypes(include="object").columns:
    mask = df[col].str.strip().str.lower().isin(SENTINEL_STRINGS)
    df.loc[mask, col] = np.nan

print(f"  Sentinel strings normalised to NaN.")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Convert numeric columns
# ─────────────────────────────────────────────────────────────
for col in ["price", "product_rating", "product_rating_count"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ─────────────────────────────────────────────────────────────
# STEP 3 — Remove impossible/extreme prices
# price <= 0   → data entry error or gift card artefact
# price > 99th pct per currency → likely a typo or display unit mismatch
# We do this per-currency so e.g. INR prices aren't cut by EUR thresholds.
# ─────────────────────────────────────────────────────────────
p99 = df.groupby("currency")["price"].transform(lambda s: s.quantile(0.99))
before = len(df)
df = df[(df["price"] > 0) & (df["price"] <= p99)].copy()
print(f"  Dropped {before - len(df):,} rows: price <= 0 or above 99th pct per currency.")

# ─────────────────────────────────────────────────────────────
# STEP 4 — Rating sanity
# IKEA ratings: 1.0–5.0. Anything outside is a scraping artefact.
# ─────────────────────────────────────────────────────────────
bad_rating = (
    df["product_rating"].notna() &
    ~df["product_rating"].between(1.0, 5.0, inclusive="both")
)
df.loc[bad_rating, "product_rating"] = np.nan
print(f"  Nullified {bad_rating.sum():,} out-of-range rating values.")

# Review count: negative or > 1M → scraper duplication artefact
bad_count = (df["product_rating_count"] < 0) | (df["product_rating_count"] > 1_000_000)
df.loc[bad_count, "product_rating_count"] = np.nan

# ─────────────────────────────────────────────────────────────
# STEP 5 — Separate "missing data" from "known zero"
# has_rating_data = 1 means we actually received a rating value (even if low).
# has_rating_data = 0 means no data at all — different business interpretation.
# Downstream scripts should not fillna(0) these blindly.
# ─────────────────────────────────────────────────────────────
df["has_rating_data"]  = df["product_rating"].notna().astype(int)
df["has_review_data"]  = df["product_rating_count"].notna().astype(int)

# ─────────────────────────────────────────────────────────────
# STEP 6 — Deduplicate listing rows
# unique_id is the row-level key. Duplicates mean the same listing
# was scraped more than once (network retry, pagination overlap).
# ─────────────────────────────────────────────────────────────
before_dedup = len(df)
df = df.drop_duplicates(subset=["unique_id"]).copy()
print(f"  Dropped {before_dedup - len(df):,} duplicate unique_id rows.")

# Drop rows with no product_id — useless for any aggregation
df = df.dropna(subset=["product_id"]).copy()

# ─────────────────────────────────────────────────────────────
# STEP 7 — Log-transform skewed raw columns
# price and product_rating_count are heavily right-skewed.
# log1p(x) = log(1 + x):
#   - safe when x = 0 (log1p(0) = 0)
#   - compresses outliers without removing them
#   - makes distributions closer to normal, which helps StandardScaler
#     work as intended in ikea_advanced_ds.py
# We store these as new columns so the original values are preserved
# for display purposes (e.g. showing "£249" not "log(250)" in charts).
# ─────────────────────────────────────────────────────────────
for col in LOG_TRANSFORM_COLS:
    if col in df.columns:
        df[f"{col}_log"] = np.log1p(df[col].fillna(0.0))

print(f"  Log1p columns added: {[c + '_log' for c in LOG_TRANSFORM_COLS]}")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
n_clean = len(df)
print(f"\nCleaning complete.")
print(f"  Raw rows:   {n_raw:,}")
print(f"  Clean rows: {n_clean:,}  ({n_raw - n_clean:,} removed, "
      f"{(n_raw - n_clean) / n_raw:.1%} of total)")
print(f"  Columns:    {len(df.columns)} (including {len(LOG_TRANSFORM_COLS)} new log columns)")
print(f"  Products:   {df['product_id'].nunique():,} unique")
print(f"  Countries:  {df['country'].nunique()}")
print(f"  Currencies: {df['currency'].nunique()}")

# Quick data-quality snapshot
missing_summary = (
    df.isnull().sum()
      .rename("missing")
      .to_frame()
      .assign(pct=lambda x: x["missing"] / n_clean)
      .query("missing > 0")
      .sort_values("pct", ascending=False)
      .head(15)
)
print(f"\nTop remaining NaN columns after cleaning:")
print(missing_summary.to_string(float_format="{:.1%}".format))

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")
print("Next step: run  python ikea_advanced_ds.py")
