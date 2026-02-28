"""
ikea_advanced_ds.py
===================
STEP 2 of 3 — Feature Engineering, Clustering & Segment Naming

Reads:   ikea_clean.csv              (from ikea_cleaner.py)
Writes:  ikea_advanced_features.csv  (product-level enriched table)

What it does:
  1. Builds a 30-feature product-centric table (market spread, price rank,
     category entropy, physical dims, engagement score, tag shares, etc.)
  2. Log-transforms skewed features, then StandardScaler
  3. Multi-method model search: KMeans, GMM, HDBSCAN across configs + seeds
  4. Composite scoring: silhouette + Davies-Bouldin + ARI stability + noise
  5. Fits final labels with the champion configuration
  6. Isolation Forest anomaly detection
  7. Rule-based segment naming from cluster centroids
  8. Saves enriched CSV for ikea_pro_viz.py
"""

from itertools import combinations
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── optional: HDBSCAN ────────────────────────────────────────────────────────
try:
    import hdbscan as hdbscan_mod
    _HAS_HDBSCAN = True
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "hdbscan"])
    import hdbscan as hdbscan_mod
    _HAS_HDBSCAN = True

pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CLEAN_PATH = Path("e:/Excel Datasets/ikea_clean.csv")
OUT_PATH   = Path("e:/Excel Datasets/ikea_advanced_features.csv")

KMEANS_K      = [4, 5, 6, 7]
GMM_K         = [4, 5, 6, 7]
HDBSCAN_CFGS  = [
    {"min_cluster_size": 120, "min_samples": 10},
    {"min_cluster_size": 220, "min_samples": 15},
    {"min_cluster_size": 320, "min_samples": 20},
]
SEEDS = [7, 21, 42]

# ─────────────────────────────────────────────────────────────────────────────
# GUARD
# ─────────────────────────────────────────────────────────────────────────────
if not CLEAN_PATH.exists():
    raise FileNotFoundError(
        f"{CLEAN_PATH} not found.\n"
        "Run  python ikea_cleaner.py  first."
    )

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading: {CLEAN_PATH}")
df = pd.read_csv(CLEAN_PATH, low_memory=False)
print(f"  Rows: {len(df):,}  |  Products: {df['product_id'].nunique():,}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _mode(s):
    m = s.mode()
    return m.iat[0] if not m.empty else s.iloc[0]


def extract_dims(s):
    """Parse measurement string → [d1, d2, d3], NaN-padded."""
    if pd.isna(s) or str(s).strip().lower() in ("none", ""):
        return [np.nan, np.nan, np.nan]
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(s))]
    return (nums + [np.nan, np.nan, np.nan])[:3]


def mean_pairwise_ari(runs):
    """Average ARI across all pairs of label runs = stability score."""
    if len(runs) < 2:
        return np.nan
    return float(np.mean([
        adjusted_rand_score(a, b) for a, b in combinations(runs, 2)
    ]))


def cluster_metrics(X, labels):
    """Silhouette + Davies-Bouldin, excluding HDBSCAN noise (-1)."""
    labels = np.asarray(labels)
    noise  = labels == -1
    mask   = ~noise if noise.any() else np.ones(len(labels), dtype=bool)
    Xev, Lev = X[mask], labels[mask]
    n_clust   = int(np.unique(Lev).size)
    if n_clust < 2 or len(Lev) < 10:
        return dict(n_clusters=n_clust, noise_share=float(noise.mean()),
                    silhouette=np.nan, davies_bouldin=np.nan)
    kw = {"sample_size": 8000, "random_state": 42} if len(Lev) > 8000 else {}
    return dict(
        n_clusters    =n_clust,
        noise_share   =float(noise.mean()),
        silhouette    =float(silhouette_score(Xev, Lev, **kw)),
        davies_bouldin=float(davies_bouldin_score(Xev, Lev)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Feature engineering ...")

# — Discount pct from string like "20%"
df["discount_pct"] = pd.to_numeric(
    df["discount"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
)

# — Binary flags
df["is_rated"]     = df["product_rating"].notna().astype(int)
df["has_discount"] = df["discount_pct"].notna().astype(int)
df["is_promoted"]  = df["sale_tag"].ne("NONE").astype(int)
df["is_top_seller"]= (df["badge"] == "TOP_SELLER").astype(int)
df["is_limited"]   = (df["badge"] == "LIMITED_EDITION").astype(int)

# — Sale tag one-hot
for tag, col in {
    "NEW_PRODUCT":           "tag__new_product",
    "NEW_LOWER_PRICE":       "tag__new_lower_price",
    "FAMILY_PRICE":          "tag__family_price",
    "TIME_RESTRICTED_OFFER": "tag__time_restricted_offer",
    "DISCONTINUED_PRODUCT":  "tag__discontinued",
}.items():
    df[col] = (df["sale_tag"] == tag).astype(int)

# — Currency-safe price rank
#   Ranks each product's price within its own currency (0-1 pct).
#   This sidesteps cross-currency distortion: ₹2000 vs €150 look
#   incomparable in raw form but are meaningful as within-market percentiles.
df["price_rank_pct"] = df.groupby("currency")["price"].rank(
    pct=True, method="average"
)

# — Top-8 category shares per product
TOP_CATS = df["main_category"].value_counts().head(8).index.tolist()
cat_dummies = pd.get_dummies(df["main_category"]).reindex(columns=TOP_CATS, fill_value=0)
cat_dummies.index = df["product_id"].values
cat_shares  = cat_dummies.groupby(cat_dummies.index).mean()
cat_shares.columns = [f"cat__{c.replace('-','_').replace(' ','_')[:20]}" for c in cat_shares.columns]
cat_shares["cat__other"] = (1.0 - cat_shares.sum(axis=1)).clip(lower=0.0)
CAT_COLS = cat_shares.columns.tolist()

# — Category entropy per product (fragmented footprint = high entropy)
_cs = cat_shares.replace(0, np.nan)
cat_entropy = -(np.log(_cs) * _cs).sum(axis=1).fillna(0.0)

# — Physical dimensions from measurement strings
print("   Extracting physical dimensions ...")
dim_raw = df.drop_duplicates("product_id")[["product_id", "product_measurements"]].copy()
dim_raw[["dim1", "dim2", "dim3"]] = pd.DataFrame(
    dim_raw["product_measurements"].apply(extract_dims).tolist(),
    index=dim_raw.index,
)

# — Aggregate to product level
prod = (
    df.groupby("product_id", as_index=False).agg(
        product_name          = ("product_name",            _mode),
        main_category         = ("main_category",           _mode),
        sub_category          = ("sub_category",            _mode),
        listing_rows          = ("unique_id",               "size"),
        market_spread         = ("country",                 "nunique"),
        currency_spread       = ("currency",                "nunique"),
        category_count        = ("main_category",           "nunique"),
        subcategory_count     = ("sub_category",            "nunique"),
        online_sellable_share = ("online_sellable",         "mean"),
        rating_presence       = ("is_rated",                "mean"),
        mean_rating           = ("product_rating",          "mean"),
        mean_rating_count     = ("product_rating_count",    "mean"),
        promoted_share        = ("is_promoted",             "mean"),
        discount_share        = ("has_discount",            "mean"),
        top_seller_share      = ("is_top_seller",           "mean"),
        limited_share         = ("is_limited",              "mean"),
        price_rank_mean       = ("price_rank_pct",          "mean"),
        price_rank_std        = ("price_rank_pct",          "std"),
        price_rank_q90        = ("price_rank_pct",          lambda s: s.quantile(0.9)),
        tag__new_product      = ("tag__new_product",        "mean"),
        tag__new_lower_price  = ("tag__new_lower_price",    "mean"),
        tag__family_price     = ("tag__family_price",       "mean"),
        tag__time_limit       = ("tag__time_restricted_offer", "mean"),
        tag__discontinued     = ("tag__discontinued",       "mean"),
        avg_price_raw         = ("price",                   "mean"),
    )
)

# — Merge dimensions + derived physical features
prod = prod.merge(dim_raw[["product_id","dim1","dim2","dim3"]], on="product_id", how="left")
prod["volume_est"]   = (prod["dim1"] * prod["dim2"] * prod["dim3"]).fillna(0.0)
prod["aspect_ratio"] = (prod["dim1"] / (prod["dim2"] + 1e-6)).fillna(0.0)

# — Merge category shares + entropy
prod = prod.merge(
    cat_shares.reset_index().rename(columns={"index":"product_id"}),
    on="product_id", how="left"
)
prod["cat_entropy"]   = prod["product_id"].map(cat_entropy).fillna(0.0)
prod["cat_max_share"] = prod[CAT_COLS].max(axis=1).fillna(0.0)

# — Global spread ratios
N_COUNTRIES  = df["country"].nunique()
N_CURRENCIES = df["currency"].nunique()
prod["market_spread_share"]   = prod["market_spread"]   / N_COUNTRIES
prod["currency_spread_share"] = prod["currency_spread"] / N_CURRENCIES

# — Impute missing rating with dataset median (keeps feature; marks presence separately)
prod["rating_imputed"]    = prod["mean_rating"].fillna(df["product_rating"].median())
prod["log_review_count"]  = np.log1p(prod["mean_rating_count"].fillna(0.0))
prod["engagement_score"]  = prod["rating_presence"] * prod["log_review_count"]
prod["price_rank_std"]    = prod["price_rank_std"].fillna(0.0)

CLUSTER_FEATURES = [
    "market_spread_share", "currency_spread_share",
    "category_count", "subcategory_count",
    "cat_entropy", "cat_max_share",
    "price_rank_mean", "price_rank_std", "price_rank_q90",
    "online_sellable_share",
    "rating_presence", "rating_imputed", "log_review_count",
    "engagement_score",
    "promoted_share", "discount_share",
    "top_seller_share", "limited_share",
    "tag__new_product", "tag__new_lower_price",
    "tag__family_price", "tag__time_limit", "tag__discontinued",
    "volume_est", "aspect_ratio",
] + CAT_COLS

feat_mat = prod[CLUSTER_FEATURES].fillna(0.0).copy()

# Auto log1p any feature with skew > 1.5 (must be non-negative)
# volume_est in particular ranges from 0 → millions without this
for col in CLUSTER_FEATURES:
    if col in feat_mat.columns and feat_mat[col].min() >= 0:
        if feat_mat[col].skew() > 1.5:
            feat_mat[col] = np.log1p(feat_mat[col])

scaler = StandardScaler()
X      = scaler.fit_transform(feat_mat)
print(f"   Products: {len(prod):,}  |  Features: {len(CLUSTER_FEATURES)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MULTI-METHOD MODEL SEARCH
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Multi-method model search ...")


def _eval(method, config, primary_param, runs, metrics):
    return dict(
        method=method, config=config, primary_param=primary_param,
        n_clusters    =float(np.nanmean([m["n_clusters"]     for m in metrics])),
        noise_share   =float(np.nanmean([m["noise_share"]    for m in metrics])),
        silhouette    =float(np.nanmean([m["silhouette"]     for m in metrics])),
        db_index      =float(np.nanmean([m["davies_bouldin"] for m in metrics])),
        stability     =mean_pairwise_ari(runs),
    )


search_rows = []

for k in KMEANS_K:
    runs, metrics = [], []
    for s in SEEDS:
        lbl = KMeans(n_clusters=k, n_init=20, random_state=s).fit_predict(X)
        runs.append(lbl); metrics.append(cluster_metrics(X, lbl))
    search_rows.append(_eval("KMeans", f"k={k}", k, runs, metrics))
    print(f"   KMeans k={k}  sil={search_rows[-1]['silhouette']:.3f}  "
          f"stab={search_rows[-1]['stability']:.3f}")

for k in GMM_K:
    runs, metrics = [], []
    for s in SEEDS:
        lbl = GaussianMixture(n_components=k, covariance_type="diag",
                              reg_covar=1e-6, random_state=s).fit_predict(X)
        runs.append(lbl); metrics.append(cluster_metrics(X, lbl))
    search_rows.append(_eval("GMM", f"components={k}", k, runs, metrics))
    print(f"   GMM   k={k}  sil={search_rows[-1]['silhouette']:.3f}  "
          f"stab={search_rows[-1]['stability']:.3f}")

rng = np.random.default_rng(0)
for cfg in HDBSCAN_CFGS:
    mcs, ms = cfg["min_cluster_size"], cfg["min_samples"]
    runs, metrics = [], []
    for _ in SEEDS:
        Xp  = X + rng.normal(0, 0.01, X.shape)
        lbl = hdbscan_mod.HDBSCAN(
            min_cluster_size=mcs, min_samples=ms,
            cluster_selection_method="eom"
        ).fit_predict(Xp)
        runs.append(lbl); metrics.append(cluster_metrics(X, lbl))
    search_rows.append(_eval("HDBSCAN", f"mcs={mcs},ms={ms}", mcs, runs, metrics))
    print(f"   HDBSCAN mcs={mcs}  sil={search_rows[-1]['silhouette']:.3f}  "
          f"stab={search_rows[-1]['stability']:.3f}")

results = pd.DataFrame(search_rows)

# Composite score (percentile ranks, so each metric is comparable)
# Minimum 4 clusters required — 2-cluster solutions (one giant blob + one tiny
# outlier group) score well on silhouette but are meaningless for business use.
valid = results[results["n_clusters"] >= 4].copy()
valid["r_sil"]   = valid["silhouette"].rank(pct=True)
valid["r_db"]    = (-valid["db_index"]).rank(pct=True)    # lower DB = better
valid["r_stab"]  = valid["stability"].rank(pct=True)
valid["r_noise"] = (-valid["noise_share"]).rank(pct=True) # less noise = better
valid["score"]   = (
    0.35 * valid["r_sil"] +
    0.25 * valid["r_db"]  +
    0.25 * valid["r_stab"]+
    0.15 * valid["r_noise"]
)
ranked   = valid.sort_values("score", ascending=False).reset_index(drop=True)
champion = ranked.iloc[0]
print(f"\n   Champion: {champion['method']} | {champion['config']}"
      f" | sil={champion['silhouette']:.3f} | stability={champion['stability']:.3f}")

print("\n   Full rankings:")
print(ranked[["method","config","n_clusters","silhouette","db_index",
              "stability","noise_share","score"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FIT FINAL LABELS
# ─────────────────────────────────────────────────────────────────────────────
def fit_champion(row, X):
    if row["method"] == "KMeans":
        return KMeans(n_clusters=int(row["primary_param"]),
                      n_init=50, random_state=42).fit_predict(X)
    if row["method"] == "GMM":
        return GaussianMixture(n_components=int(row["primary_param"]),
                               covariance_type="diag", reg_covar=1e-6,
                               random_state=42).fit_predict(X)
    if row["method"] == "HDBSCAN":
        parts = dict(item.split("=") for item in row["config"].split(","))
        return hdbscan_mod.HDBSCAN(
            min_cluster_size=int(parts["mcs"]),
            min_samples=int(parts["ms"]),
            cluster_selection_method="eom",
        ).fit_predict(X)
    raise ValueError(row["method"])

prod["cluster_id"] = fit_champion(champion, X)
prod["cluster_method"] = champion["method"]
prod["cluster_config"] = champion["config"]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Anomaly detection (Isolation Forest) ...")
iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
prod["is_anomaly"] = (iso.fit_predict(X) == -1).astype(int)
print(f"   Anomalies: {prod['is_anomaly'].sum():,}  "
      f"({prod['is_anomaly'].mean():.1%} of products)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SEGMENT NAMING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Naming segments ...")


def _name_cluster(cid, centroid):
    if cid == -1:
        return "Outliers / Noise"
    breadth  = centroid.get("market_spread_share", 0)
    price    = centroid.get("price_rank_mean", 0.5)
    engage   = centroid.get("engagement_score", 0)
    promo    = centroid.get("promoted_share", 0)
    top_sell = centroid.get("top_seller_share", 0)
    discont  = centroid.get("tag__discontinued", 0)
    new_prod = centroid.get("tag__new_product", 0)
    limited  = centroid.get("limited_share", 0)

    # Strongest categorical signals override geography/price rules
    if limited  > 0.15: return "Limited Edition Exclusives"
    if discont  > 0.10: return "End-of-Life / Discontinued"
    if new_prod > 0.15: return "New Arrivals Pipeline"
    if top_sell > 0.25: return "Top Sellers (Proven Hits)"
    # Geography + price tier
    if breadth > 0.70 and price > 0.65: return "Global Premium Flagships"
    if breadth > 0.70 and price < 0.40: return "Global Value Champions"
    if breadth > 0.70:                  return "Well-Travelled Mid-Range"
    if breadth < 0.25 and price > 0.60: return "Regional Luxury Niche"
    if breadth < 0.25 and promo > 0.20: return "Locally Promoted Specials"
    if breadth < 0.25:                  return "Single-Market Exclusives"
    # Engagement signal
    if engage > 0.50 and price > 0.55:  return "Reviewed Premium Picks"
    if engage > 0.50:                   return "Community Favourites"
    return "Standard Portfolio"


cluster_ids = sorted(prod["cluster_id"].unique(), key=lambda x: (x == -1, x))
profile_rows = []
for cid in cluster_ids:
    sub      = prod[prod["cluster_id"] == cid]
    centroid = sub[CLUSTER_FEATURES].mean().to_dict()
    name     = _name_cluster(cid, centroid)
    profile_rows.append({
        "cluster_id":   cid,
        "segment_name": name,
        "n_products":   len(sub),
        **{k: centroid[k] for k in [
            "market_spread_share", "price_rank_mean", "top_seller_share",
            "engagement_score", "promoted_share", "limited_share",
            "tag__discontinued", "cat_entropy",
        ]},
    })

cluster_profile = pd.DataFrame(profile_rows)
id_to_name = dict(zip(cluster_profile["cluster_id"], cluster_profile["segment_name"]))
prod["segment_name"] = prod["cluster_id"].map(id_to_name)

print("   Segments found:")
for _, row in cluster_profile.iterrows():
    print(f"     [{row['cluster_id']:>3}]  {row['segment_name']:<35} "
          f"{row['n_products']:,} products")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — SAVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Saving ...")

SAVE_COLS = [
    "product_id", "product_name", "main_category", "sub_category",
    "cluster_id", "segment_name", "cluster_method", "cluster_config",
    "is_anomaly",
    "market_spread", "market_spread_share",
    "avg_price_raw", "price_rank_mean", "price_rank_std",
    "rating_presence",             # needed by pro_viz radar chart
    "rating_imputed", "mean_rating", "mean_rating_count",
    "log_review_count", "engagement_score",
    "top_seller_share", "promoted_share", "discount_share",
    "limited_share", "tag__new_product", "tag__new_lower_price",
    "tag__discontinued",
    "volume_est", "aspect_ratio",
    "cat_entropy", "cat_max_share",
    "online_sellable_share",
    "listing_rows",
] + CAT_COLS

prod[SAVE_COLS].to_csv(OUT_PATH, index=False)
cluster_profile.to_csv("e:/Excel Datasets/ikea_cluster_profiles.csv", index=False)

print(f"   Enriched features → {OUT_PATH}")
print("   Cluster profiles  → e:/Excel Datasets/ikea_cluster_profiles.csv")
print("\nValidation scores:")
print(f"   Silhouette:      {champion['silhouette']:.4f}")
print(f"   Davies-Bouldin:  {champion['db_index']:.4f}")
print(f"   ARI Stability:   {champion['stability']:.4f}")
print(f"   Noise share:     {champion['noise_share']:.1%}")
print("\nNext step: run  python ikea_pro_viz.py")
