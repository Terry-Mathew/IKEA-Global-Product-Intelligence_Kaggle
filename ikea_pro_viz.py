"""
ikea_pro_viz.py
===============
STEP 3 of 3 — Predictive Modelling & Interactive Visualisations

Reads:   ikea_advanced_features.csv  (from ikea_advanced_ds.py)
Writes:  ikea_dashboard.html          (main deliverable — opens in browser)
         ikea_product_galaxy.html
         ikea_umap_interactive.html
         ikea_cluster_radar.html
         ikea_model_search.png
         ikea_feature_importance.png

What it does:
  1. XGBoost model to predict top-seller probability per product
  2. UMAP projection (full dataset, global structure)
  3. t-SNE projection (stratified sample, local structure check)
  4. Static overview charts with the warm paper palette
  5. Interactive Plotly: 3D galaxy, UMAP scatter, cluster radar
  6. Single self-contained HTML dashboard with a navigation bar
"""

import importlib
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")


def _ensure(import_name, pip_name=None):
    try:
        return importlib.import_module(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user",
                               pip_name or import_name])
        return importlib.import_module(import_name)


umap_mod = _ensure("umap", "umap-learn")

try:
    from xgboost import XGBRegressor
except ImportError:
    _ensure("xgboost")
    from xgboost import XGBRegressor

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — warm paper palette
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "ink":   "#102A43",
    "slate": "#486581",
    "paper": "#F7F5EF",
    "line":  "#D9D4C7",
    "teal":  "#117A8B",
    "coral": "#D96C4B",
    "gold":  "#E6A23C",
    "moss":  "#5C7C4F",
    "berry": "#A33E5B",
}

CLUSTER_COLORS = [
    "#117A8B","#D96C4B","#E6A23C","#5C7C4F",
    "#A33E5B","#486581","#7B8794","#2E86C1",
    "#884EA0","#17A589",
]

PLOTLY_BASE = dict(
    paper_bgcolor=PALETTE["paper"],
    plot_bgcolor =PALETTE["paper"],
    font         =dict(family="Inter, Arial, sans-serif", color=PALETTE["ink"]),
    title_font   =dict(size=20, color=PALETTE["ink"]),
    legend       =dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin       =dict(l=60, r=30, t=80, b=60),
)

sns.set_theme(
    context="talk", style="whitegrid",
    rc={
        "figure.facecolor": PALETTE["paper"],
        "axes.facecolor":   PALETTE["paper"],
        "grid.color":       PALETTE["line"],
        "axes.edgecolor":   "#C7C0AF",
        "text.color":       PALETTE["ink"],
        "axes.labelcolor":  PALETTE["ink"],
    },
)
mpl.rcParams.update({
    "axes.titleweight": "bold",
    "axes.titlesize":   15,
    "axes.labelsize":   12,
    "figure.dpi":       120,
    "savefig.bbox":     "tight",
})


def human_fmt(v):
    if pd.isna(v): return "NA"
    v = float(v)
    for s, t in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(v) >= t:
            x = v / t
            return f"{x:.0f}{s}" if x == int(x) else f"{x:.1f}{s}"
    return f"{v:.0f}" if v == int(v) else f"{v:.1f}"


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    for sp in ["top","right","left"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    if title:  ax.set_title(title, loc="left", pad=10)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)


# ─────────────────────────────────────────────────────────────────────────────
# GUARD + LOAD
# ─────────────────────────────────────────────────────────────────────────────
FEAT_PATH = Path("e:/Excel Datasets/ikea_advanced_features.csv")
PROF_PATH = Path("e:/Excel Datasets/ikea_cluster_profiles.csv")

if not FEAT_PATH.exists():
    raise FileNotFoundError(
        f"{FEAT_PATH} not found.\n"
        "Run  python ikea_advanced_ds.py  first."
    )

print(f"Loading: {FEAT_PATH}")
prod = pd.read_csv(FEAT_PATH, low_memory=False)
print(f"  Products: {len(prod):,}  |  Segments: {prod['segment_name'].nunique()}")

cluster_ids = sorted(prod["cluster_id"].unique(), key=lambda x: (x == -1, x))
non_noise   = [c for c in cluster_ids if c != -1]

id_to_color = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(non_noise)}
if -1 in cluster_ids:
    id_to_color[-1] = "#AAAAAA"

id_to_name  = dict(zip(prod["cluster_id"], prod["segment_name"]))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — XGBOOST POPULARITY PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Training XGBoost popularity predictor ...")

# Numeric-only features (exclude identifiers + target)
EXCLUDE = {"product_id","product_name","main_category","sub_category",
           "segment_name","cluster_method","cluster_config","top_seller_share"}
ML_FEATS = [c for c in prod.select_dtypes(include=np.number).columns
            if c not in EXCLUDE and c != "top_seller_share"]

TARGET = "top_seller_share"
X_ml   = prod[ML_FEATS].fillna(0.0).values
y_ml   = prod[TARGET].fillna(0.0).values

X_tr, X_te, y_tr, y_te = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)

xgb = XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", random_state=42, n_jobs=-1,
)
xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
preds = xgb.predict(X_te)
r2   = r2_score(y_te, preds)
rmse = mean_squared_error(y_te, preds) ** 0.5
print(f"   R² = {r2:.4f}  |  RMSE = {rmse:.4f}")

prod["predicted_top_seller_pct"] = xgb.predict(X_ml)

feat_imp = (
    pd.DataFrame({"feature": ML_FEATS, "importance": xgb.feature_importances_})
    .sort_values("importance", ascending=False).head(12).reset_index(drop=True)
)
print("   Top 6 drivers:")
for _, r in feat_imp.head(6).iterrows():
    print(f"     {r['feature']:<35} {r['importance']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — UMAP + t-SNE PROJECTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Dimensionality reduction ...")

feat_cols = [c for c in prod.select_dtypes(include=np.number).columns
             if c not in {"product_id","cluster_id","is_anomaly",
                          "predicted_top_seller_pct","umap_x","umap_y"}]
X_proj = prod[feat_cols].fillna(0.0).values

# UMAP — run on full dataset, preserves global structure
reducer  = umap_mod.UMAP(n_neighbors=35, min_dist=0.08, metric="euclidean", random_state=42)
emb      = reducer.fit_transform(X_proj)
prod["umap_x"] = emb[:, 0]
prod["umap_y"] = emb[:, 1]
print("   UMAP done.")

# t-SNE — stratified sample to check local structure
N_TSNE = min(3500, len(prod))
quota  = max(40, N_TSNE // max(1, len(cluster_ids)))
parts  = [sub.sample(min(len(sub), quota), random_state=42)
          for _, sub in prod.groupby("cluster_id")]
tsne_df = pd.concat(parts).drop_duplicates("product_id")
rem = N_TSNE - len(tsne_df)
if rem > 0:
    rest    = prod[~prod["product_id"].isin(tsne_df["product_id"])]
    tsne_df = pd.concat([tsne_df, rest.sample(min(rem, len(rest)), random_state=42)])

tsne_idx = tsne_df.index.to_numpy()
tsne_emb = TSNE(
    n_components=2, perplexity=40, learning_rate="auto",
    init="pca", max_iter=900, random_state=42,
).fit_transform(X_proj[tsne_idx])
tsne_df = tsne_df.reset_index(drop=True)
tsne_df["tsne_x"] = tsne_emb[:, 0]
tsne_df["tsne_y"] = tsne_emb[:, 1]
prod = prod.merge(tsne_df[["product_id","tsne_x","tsne_y"]], on="product_id", how="left")
print("   t-SNE done.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — STATIC CHARTS (seaborn / matplotlib, warm palette)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Static charts ...")

# Chart A: Segment sizes + XGBoost feature importance
fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=PALETTE["paper"])

seg_counts = (
    prod.groupby(["cluster_id","segment_name"])
        .size().reset_index(name="products")
        .sort_values("products", ascending=True)
)
bar_colors = [id_to_color.get(c, "#888") for c in seg_counts["cluster_id"]]
axes[0].barh(seg_counts["segment_name"], seg_counts["products"],
             color=bar_colors, edgecolor="white")
for _, r in seg_counts.iterrows():
    axes[0].text(r["products"] + 15, r["segment_name"],
                 human_fmt(r["products"]), va="center", fontsize=9)
style_ax(axes[0], title="Segment Size Distribution", xlabel="Products")

axes[1].barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1],
             color=PALETTE["teal"], edgecolor="white")
style_ax(axes[1], title="Top Drivers of Top-Seller Probability (XGBoost)",
         xlabel="Feature Importance")
plt.tight_layout()
plt.savefig("e:/Excel Datasets/ikea_model_search.png", dpi=130)
plt.close()
print("   Saved: ikea_model_search.png")

# Chart B: UMAP (static)
fig, ax = plt.subplots(figsize=(12, 8), facecolor=PALETTE["paper"])
for cid in non_noise:
    sub = prod[prod["cluster_id"] == cid]
    ax.scatter(sub["umap_x"], sub["umap_y"], c=id_to_color[cid],
               s=8, alpha=0.6, linewidths=0, label=id_to_name.get(cid, str(cid)))
if -1 in cluster_ids:
    sub = prod[prod["cluster_id"] == -1]
    ax.scatter(sub["umap_x"], sub["umap_y"], c="#CCCCCC", s=4,
               alpha=0.3, linewidths=0, label="Noise")
style_ax(ax, title="UMAP Projection — Products by Segment",
         xlabel="UMAP 1", ylabel="UMAP 2")
ax.legend(frameon=False, fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("e:/Excel Datasets/ikea_umap_static.png", dpi=130)
plt.close()
print("   Saved: ikea_umap_static.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — INTERACTIVE PLOTLY CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Interactive Plotly charts ...")

# — 3D Product Galaxy
galaxy = prod.dropna(subset=["avg_price_raw","rating_imputed","volume_est"])
galaxy = galaxy[galaxy["volume_est"] > 0]
galaxy = galaxy[galaxy["volume_est"] < galaxy["volume_est"].quantile(0.97)]
galaxy = galaxy.sample(min(6000, len(galaxy)), random_state=42)

fig_3d = px.scatter_3d(
    galaxy,
    x="avg_price_raw", y="rating_imputed", z="volume_est",
    color="segment_name",
    color_discrete_sequence=CLUSTER_COLORS,
    size="engagement_score", size_max=18, opacity=0.78,
    hover_name="product_name",
    hover_data={
        "segment_name": True, "market_spread": True,
        "top_seller_share": ":.2f", "is_anomaly": True,
        "avg_price_raw": ":.1f", "rating_imputed": ":.2f",
        "predicted_top_seller_pct": ":.2f",
        "volume_est": False, "engagement_score": False,
    },
    title="🪑 IKEA Product Galaxy — Price × Rating × Size",
    labels={
        "avg_price_raw":  "Avg Price (local currency)",
        "rating_imputed": "Customer Rating",
        "volume_est":     "Physical Volume (cm³)",
        "segment_name":   "Segment",
    },
)
fig_3d.update_layout(
    **PLOTLY_BASE,
    scene=dict(
        xaxis=dict(backgroundcolor=PALETTE["paper"], gridcolor=PALETTE["line"]),
        yaxis=dict(backgroundcolor=PALETTE["paper"], gridcolor=PALETTE["line"]),
        zaxis=dict(backgroundcolor=PALETTE["paper"], gridcolor=PALETTE["line"]),
    ),
    scene_camera=dict(eye=dict(x=1.35, y=1.35, z=0.6)),
)
fig_3d.write_html("e:/Excel Datasets/ikea_product_galaxy.html", include_plotlyjs="cdn")
print("   Saved: ikea_product_galaxy.html")

# — UMAP interactive
fig_umap = px.scatter(
    prod, x="umap_x", y="umap_y",
    color="segment_name",
    color_discrete_sequence=CLUSTER_COLORS,
    hover_name="product_name",
    hover_data={"segment_name": True, "market_spread": True,
                "top_seller_share": ":.2f", "is_anomaly": True,
                "predicted_top_seller_pct": ":.2f",
                "umap_x": False, "umap_y": False},
    title="UMAP Projection — All Products Coloured by Segment",
    opacity=0.65,
)
fig_umap.update_traces(marker=dict(size=5, line=dict(width=0)))
fig_umap.update_layout(**PLOTLY_BASE, xaxis_title="UMAP 1", yaxis_title="UMAP 2")
fig_umap.write_html("e:/Excel Datasets/ikea_umap_interactive.html", include_plotlyjs="cdn")
print("   Saved: ikea_umap_interactive.html")

# — Cluster Radar (per-cluster normalised)
RADAR_FEATS_WANTED  = ["market_spread_share","cat_entropy","price_rank_mean",
                       "rating_presence","promoted_share","top_seller_share",
                       "limited_share","online_sellable_share"]
RADAR_LABELS_WANTED = ["Market Reach","Category Mix","Price Tier","Rating Presence",
                       "Promo Rate","Top Seller","Limited Edition","Online Sellable"]

# Only keep features that are actually in the CSV
_pairs = [(f, l) for f, l in zip(RADAR_FEATS_WANTED, RADAR_LABELS_WANTED)
          if f in prod.columns]
RADAR_FEATS  = [p[0] for p in _pairs]
RADAR_LABELS = [p[1] for p in _pairs]
print(f"   Radar features available: {RADAR_FEATS}")

seg_means = (
    prod.groupby(["cluster_id","segment_name"])[RADAR_FEATS].mean().reset_index()
)
non_noise_segs = seg_means[seg_means["cluster_id"] != -1].copy()

def _norm_row(r):
    lo, hi = r[RADAR_FEATS].min(), r[RADAR_FEATS].max()
    return (r[RADAR_FEATS] - lo) / (hi - lo + 1e-9)

n_panels = len(non_noise_segs)
n_cols   = min(3, n_panels)
n_rows   = math.ceil(n_panels / n_cols)

fig_radar = make_subplots(
    rows=n_rows, cols=n_cols,
    specs=[[{"type": "polar"}] * n_cols for _ in range(n_rows)],
    subplot_titles=non_noise_segs["segment_name"].tolist(),
)
for idx, row in non_noise_segs.iterrows():
    ri = (list(non_noise_segs.index).index(idx)) // n_cols + 1
    ci = (list(non_noise_segs.index).index(idx)) % n_cols  + 1
    vals  = _norm_row(row).tolist() + [_norm_row(row).iloc[0]]
    lbls  = RADAR_LABELS + [RADAR_LABELS[0]]
    color = id_to_color.get(row["cluster_id"], "#888888")
    
    # Convert hex to rgba() string because Plotly rejects 8-digit hex codes
    r, g, b, _ = mpl.colors.to_rgba(color)
    fillcolor_rgba = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.3)"

    fig_radar.add_trace(
        go.Scatterpolar(
            r=vals, theta=lbls, fill="toself",
            fillcolor=fillcolor_rgba,
            line=dict(color=color, width=2.2),
            name=row["segment_name"], showlegend=False,
        ),
        row=ri, col=ci,
    )

fig_radar.update_layout(
    **PLOTLY_BASE, height=350 * n_rows,
    title_text="Cluster Radar Profiles (per-cluster normalised)",
)
fig_radar.write_html("e:/Excel Datasets/ikea_cluster_radar.html", include_plotlyjs="cdn")
print("   Saved: ikea_cluster_radar.html")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FULL HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Building dashboard ...")

div_galaxy = pio.to_html(fig_3d,    full_html=False, include_plotlyjs=False, div_id="galaxy")
div_umap   = pio.to_html(fig_umap,  full_html=False, include_plotlyjs=False, div_id="umap")
div_radar  = pio.to_html(fig_radar, full_html=False, include_plotlyjs=False, div_id="radar")

seg_table_rows = ""
for _, r in non_noise_segs.iterrows():
    color = id_to_color.get(r["cluster_id"], "#888")
    n_prod = prod[prod["cluster_id"] == r["cluster_id"]].shape[0]
    seg_table_rows += (
        f"<tr>"
        f"<td><b>{int(r['cluster_id'])}</b></td>"
        f"<td><span class='chip' style='background:{color}'>{r['segment_name']}</span></td>"
        f"<td>{human_fmt(n_prod)}</td>"
        f"<td>{r.get('market_spread_share',0):.1%}</td>"
        f"<td>{r.get('price_rank_mean',0):.1%}</td>"
        f"<td>{r.get('top_seller_share',0):.1%}</td>"
        f"</tr>"
    )

anomaly_rows = ""
top_anomalies = (
    prod[prod["is_anomaly"]==1]
    .sort_values("predicted_top_seller_pct", ascending=False)
    .head(20)
)
for _, r in top_anomalies.iterrows():
    color = id_to_color.get(r["cluster_id"], "#888")
    anomaly_rows += (
        f"<tr>"
        f"<td>{str(r['product_name'])[:55]}</td>"
        f"<td><span class='chip' style='background:{color}'>{r['segment_name']}</span></td>"
        f"<td>{int(r.get('market_spread', 0))}</td>"
        f"<td>{r.get('price_rank_mean', 0):.1%}</td>"
        f"<td>{r.get('predicted_top_seller_pct', 0):.1%}</td>"
        f"</tr>"
    )

DASH = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>IKEA Product Intelligence Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
  :root {{
    --ink:   {PALETTE['ink']};
    --paper: {PALETTE['paper']};
    --teal:  {PALETTE['teal']};
    --line:  {PALETTE['line']};
    --slate: {PALETTE['slate']};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--paper); font-family: Inter, Arial, sans-serif; color: var(--ink); }}
  header {{
    background: var(--ink); color: #fff;
    padding: 18px 32px; display: flex; align-items: center; gap: 16px;
  }}
  header h1 {{ font-size: 1.4rem; font-weight: 700; }}
  header p  {{ font-size: 0.8rem; opacity: 0.6; margin-top: 2px; }}
  .badge {{
    background: {PALETTE['teal']}; color: #fff;
    border-radius: 4px; padding: 3px 10px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: .5px;
  }}
  nav {{
    background: #fff; border-bottom: 2px solid var(--line);
    display: flex; padding: 0 32px;
    position: sticky; top: 0; z-index: 100;
  }}
  nav button {{
    background: none; border: none;
    border-bottom: 3px solid transparent;
    color: var(--slate); cursor: pointer;
    font-size: 0.9rem; font-weight: 500; padding: 14px 20px;
    transition: all .18s;
  }}
  nav button:hover  {{ color: var(--ink); }}
  nav button.active {{ color: var(--teal); border-bottom-color: var(--teal); font-weight: 700; }}
  .panel {{ display: none; padding: 24px 32px; }}
  .panel.active {{ display: block; }}
  .kpis {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px; margin-bottom: 22px;
  }}
  .kpi {{
    background: #fff; border: 1px solid var(--line);
    border-radius: 10px; padding: 14px 16px;
  }}
  .kpi .val {{ font-size: 1.75rem; font-weight: 700; color: var(--teal); }}
  .kpi .lbl {{ font-size: 0.75rem; color: var(--slate); margin-top: 3px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.83rem; margin-top: 12px; }}
  th {{ background: var(--ink); color: #fff; padding: 8px 12px; text-align: left; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid var(--line); }}
  tr:hover td {{ background: #eee8dc; }}
  .chip {{
    display: inline-block; border-radius: 4px;
    padding: 2px 8px; font-size: 0.75rem; font-weight: 600; color: #fff;
  }}
  h2 {{ font-size: 0.95rem; font-weight: 700; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>🪑 IKEA Product Intelligence Dashboard</h1>
    <p>Cluster analysis · Anomaly detection · Popularity prediction</p>
  </div>
  <span class="badge">ML Pipeline v3</span>
</header>

<nav>
  <button class="active" onclick="show('overview',this)">📊 Overview</button>
  <button onclick="show('galaxy',this)">🌌 Product Galaxy</button>
  <button onclick="show('umap',this)">🗺 UMAP</button>
  <button onclick="show('radar',this)">📡 Radar</button>
  <button onclick="show('anomalies',this)">⚠️ Anomalies</button>
</nav>

<div id="overview" class="panel active">
  <div class="kpis">
    <div class="kpi"><div class="val">{human_fmt(len(prod))}</div><div class="lbl">Unique Products</div></div>
    <div class="kpi"><div class="val">{len(non_noise)}</div><div class="lbl">Segments Found</div></div>
    <div class="kpi"><div class="val">{human_fmt(prod['is_anomaly'].sum())}</div><div class="lbl">Anomalies Flagged</div></div>
    <div class="kpi"><div class="val">{r2:.3f}</div><div class="lbl">Predictor R²</div></div>
    <div class="kpi"><div class="val">{prod['cluster_method'].iloc[0]}</div><div class="lbl">Champion Model</div></div>
    <div class="kpi"><div class="val">{prod['cluster_config'].iloc[0]}</div><div class="lbl">Config</div></div>
  </div>
  <h2>Segment Summary</h2>
  <table>
    <tr><th>ID</th><th>Segment</th><th>Products</th>
        <th>Market Spread</th><th>Avg Price Rank</th><th>Top Seller %</th></tr>
    {seg_table_rows}
  </table>
</div>

<div id="galaxy"    class="panel">{div_galaxy}</div>
<div id="umap"      class="panel">{div_umap}</div>
<div id="radar"     class="panel">{div_radar}</div>

<div id="anomalies" class="panel">
  <h2>Top Anomalies by Predicted Top-Seller Probability</h2>
  <table>
    <tr><th>Product Name</th><th>Segment</th><th>Market Spread</th>
        <th>Price Rank</th><th>Predicted Top-Seller</th></tr>
    {anomaly_rows}
  </table>
</div>

<script>
function show(id, btn) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body>
</html>"""

Path("e:/Excel Datasets/ikea_dashboard.html").write_text(DASH, encoding="utf-8")
print("   Saved: ikea_dashboard.html")

print("\n✅ All done.")
print(f"   Open  e:/Excel Datasets/ikea_dashboard.html  in your browser.")
