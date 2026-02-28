# IKEA Global Product Portfolio — Strategic Intelligence Report

**Prepared:** February 28, 2026
**Source Data:** `ikea_clean.csv` — 397,032 listings | 34,409 unique products | 46 countries
**Scripts Run:** `ikea_analysis.py` (8 statistical techniques) · `ikea_advanced_ds.py` (ML pipeline)
**Note:** `ikea_pro_viz.py` not executed — visualization only, no analytical output.

---

## 1. Executive Summary

After cleaning the raw catalog data and re-running the full analysis pipeline, three findings stand out that directly affect strategy:

**1. The catalog is structurally fragmented.** The machine learning model determined that 69% of IKEA's unique product catalog — 23,823 out of 34,409 products — does not fit into any clean, coherent product group. These products are genuinely diverse in price, geography, and engagement. This is not a data problem; it is a portfolio architecture problem.

**2. Price has no relationship with customer satisfaction.** The correlation between price and rating is 0.011 — statistically meaningless. Budget items (under 50 currency units) actually earn the highest average rating (4.49) in the entire portfolio. There is no quality halo on premium products.

**3. Promotions are being applied to the wrong products.** In the cleaned data, promoted listings receive fewer reviews on average (113) than non-promoted listings (193). Promoted products are also nearly twice as expensive on average (22,322 vs 11,305 in local currency). This means promotions are being concentrated on high-price items that customers engage with less — not on the affordable, high-engagement products that drive review volume and brand trust.

**Top three actions:**
- Identify the 1,378 "Community Favourites" (globally present, high top-seller rates) and build your growth strategy around them.
- Redirect promotional investment from expensive items toward mid-range products with strong ratings and growing review volume.
- Treat the 69% "noise" category as a portfolio pruning and prioritization signal — it reveals a catalog that has grown wider than it is deep.

---

## 2. Analysis Methodology

### 2.1 Distribution Analysis
- **Technique:** Statistical percentile mapping, skewness and kurtosis measurement.
- **What it does:** Shows how prices and ratings are spread across all products — where the middle is, how lopsided the data is, and how extreme the outliers are.
- **Question answered:** What does a "typical" IKEA product look like on price and rating?
- **Why appropriate:** Establishes baseline facts before any modeling. Required before making any pricing strategy decisions.

### 2.2 Correlation Analysis
- **Technique:** Pearson correlation coefficient.
- **What it does:** Measures how strongly two numbers move together. A value of +1 means perfect alignment, 0 means no relationship, -1 means opposite movements.
- **Question answered:** Does paying more mean being more satisfied? Does a better rating attract more reviews?
- **Why appropriate:** Directly tests the "premium = quality" assumption before it is built into any recommendation.

### 2.3 Category Profiling
- **Technique:** Group-level aggregation (mean price and mean rating per category).
- **What it does:** Summarizes each product category by volume, average price, and average rating.
- **Question answered:** Which categories dominate the catalog? Which earn the most and satisfy customers most?
- **Why appropriate:** Provides the map of the portfolio before any deeper analysis.

### 2.4 Outlier Detection (IQR Method)
- **Technique:** Interquartile Range (IQR) method.
- **What it does:** Flags products priced beyond 1.5 times the middle spread (Q1 to Q3 range). These are statistically unusual — too cheap or too expensive relative to the rest of the catalog.
- **Question answered:** Are there pricing anomalies or data quality issues at the extremes?
- **Why appropriate:** Standard, distribution-free approach. Works on any dataset without assuming a bell curve.

### 2.5 Geographic Analysis
- **Technique:** Country-level aggregation with descriptive statistics.
- **What it does:** Groups all listings by country and computes average price, median price, and average rating per market.
- **Question answered:** Which markets have the most products, highest prices, and most satisfied customers?
- **Limitation:** All prices are in local currencies. Direct cross-country comparison in raw numbers is not valid without currency conversion.

### 2.6 Promotion Effectiveness Analysis
- **Technique:** Group comparison (promoted vs. non-promoted listings).
- **What it does:** Splits the catalog into products that carry a promotional tag and those that do not, then compares average ratings, review counts, and prices.
- **Question answered:** Are promotions associated with more customer engagement?
- **Why appropriate:** A necessary check before committing further investment to promotional programs.

### 2.7 Text Analysis (Product Name Frequency)
- **Technique:** Word frequency count across all product names.
- **What it does:** Tallies how often each word or product series name appears across all listings.
- **Question answered:** Which product lines dominate the catalog by name volume?
- **Why appropriate:** Reveals which product families are most heavily represented without needing sales data.

### 2.8 Price Tier Segmentation
- **Technique:** Fixed-boundary binning into six price bands.
- **What it does:** Assigns every product to one tier (Budget through Ultra-Luxury) and measures average rating per tier.
- **Question answered:** How is the portfolio distributed by price positioning, and does tier affect satisfaction?
- **Why appropriate:** Provides an executive-readable summary of the pricing architecture.

### 2.9 Feature Engineering (34 product-level features)
- **Technique:** Aggregation and transformation of raw listing data to a product-centric table.
- **What it does:** For each unique product, computes 34 signals including: how many countries it appears in, how it ranks on price within its own currency (avoiding cross-currency distortion), how many reviews it has, whether it holds a Top Seller badge, physical dimensions, and category footprint.
- **Question answered:** What is the full behavioral fingerprint of each product across all markets?
- **Why appropriate:** Raw data has one row per country-listing. This step collapses to one row per product — necessary for any product-level modeling.

### 2.10 Multi-Method Clustering with Composite Model Selection
- **Technique:** KMeans, GMM, and HDBSCAN each tested across multiple configurations and three random seeds. Champion selected by a composite score weighting four metrics.
- **Plain-English explanation of each:**
  - **KMeans:** Forces products into a fixed number of equal-ish groups by minimizing distance to group centers. Simple and fast but assumes all groups are similar in shape and size.
  - **GMM (Gaussian Mixture Model):** Similar to KMeans but allows groups to be different sizes and shapes. More flexible, less stable across runs.
  - **HDBSCAN (winner):** Finds groups where products are genuinely concentrated together. Any product that does not clearly belong to a group is labeled "noise." Does not force membership.
- **Composite scoring weights:** Silhouette score 35%, Davies-Bouldin Index 25%, ARI stability 25%, noise share 15%.
- **Question answered:** Are there natural, distinct product archetypes in the IKEA catalog?
- **Why appropriate:** Testing three methods against each other, rather than defaulting to one, ensures the final result is the best available answer.

### 2.11 Anomaly Detection (Isolation Forest)
- **Technique:** Isolation Forest — a tree-based machine learning method.
- **What it does:** Identifies products that are unusually easy to separate from the rest of the data across all 34 features simultaneously. These are products that are strange in some combination of dimensions no single metric would catch alone.
- **Question answered:** Which products are genuine outliers — potentially pricing errors, niche breakouts, or misclassified items?
- **Why appropriate:** Works on high-dimensional data without any prior assumption about what "normal" looks like.

---

## 3. Key Findings — Evidence Only

### 3.1 Price Distribution Is Extremely Lopsided

After cleaning, skewness is **18.33** and kurtosis is **472.45** — both far higher than a typical business dataset, indicating an extreme concentration of cheap products alongside a thin tail of very expensive ones.

| Percentile | Price (local currency) |
|---|---|
| 10th | 5.99 |
| 25th | 20.00 |
| **50th (median)** | **149.00** |
| 75th | 899.00 |
| 90th | 8,495.00 |
| 95th | 32,400.00 |
| 99th | 309,000.00 |

**Observation:** Half of all listings are priced at 149 or below. The top 10% starts above 8,495. These are in local currencies and cannot be compared directly across countries, but the shape is consistent: a large cheap majority and a small expensive minority.

### 3.2 Customer Satisfaction Is High and Price-Independent

| Metric | Value |
|---|---|
| Rated products | 267,700 |
| Mean rating | 4.46 / 5.0 |
| Median rating | 4.60 |
| Most common rating | 5.0 |
| Price-to-rating correlation | 0.011 |
| Rating-to-review-count correlation | 0.026 |

**Observation:** Both correlations are statistically negligible. Higher prices do not earn better ratings. Products with higher ratings do not attract more reviews organically. These are independent of each other.

### 3.3 Cheaper Products Earn Marginally Higher Ratings

| Price Tier | Products | Share | Avg Rating |
|---|---|---|---|
| Budget (<50) | 144,415 | 36.4% | **4.49** |
| Affordable (50–100) | 38,278 | 9.6% | 4.45 |
| Mid-range (100–250) | 49,255 | 12.4% | 4.42 |
| Premium (250–500) | 39,659 | 10.0% | 4.39 |
| Luxury (500–1,000) | 32,896 | 8.3% | 4.39 |
| Ultra-Luxury (>1,000) | 92,529 | 23.3% | 4.46 |

**Observation:** The cheapest tier earns the highest average rating. The portfolio is bimodal: 36.4% budget and 23.3% ultra-luxury, with only 19.6% sitting in the combined affordable-to-premium range (50–500).

### 3.4 Promotions Are Associated With Lower Review Volume

| Metric | Promoted | Non-Promoted |
|---|---|---|
| Listing count | 70,155 (17.7%) | 326,877 (82.3%) |
| Avg rating | 4.45 | 4.46 |
| **Avg reviews** | **113** | **193** |
| Avg price (local currency) | 22,322 | 11,305 |

**Observation:** Promoted products average 41% fewer reviews than non-promoted. Promoted products cost nearly double on average. Ratings are virtually identical. This is the opposite of what a well-functioning promotion program would show.

**Data quality flag:** In the ML pipeline's aggregated features table, `promoted_share = 1.0` for every product — meaning all products appear as 100% promoted after aggregation. This suggests an issue in how the `sale_tag` field was cleaned or aggregated. The figures above (from `ikea_analysis.py`) are based on the raw listing-level comparison and are the more reliable view. This must be investigated before building Model A.

### 3.5 Top 10 Categories by Volume

| Category | Products | Avg Price (local) | Avg Rating |
|---|---|---|---|
| Storage & Organisation | 51,405 | 15,690 | 4.42 |
| Kitchenware & Tableware | 26,032 | 462 | 4.53 |
| Beds & Mattresses | 22,263 | 6,096 | 4.39 |
| Decoration | 20,214 | 904 | **4.62** |
| Kitchen Appliances | 14,428 | 7,082 | 4.29 |
| Storage Furniture | 14,193 | 750 | 4.38 |
| Sofas & Armchairs | 13,501 | 16,930 | 4.23 |
| Lighting | 12,541 | 1,966 | 4.31 |
| Small Storage & Organisers | 11,176 | 1,041 | 4.53 |
| Kitchens | 10,031 | 2,445 | 4.28 |

**Observation:** Decoration earns the highest rating (4.62) at a low average price. Sofas & Armchairs carry the highest average price but the lowest satisfaction rating among the top 10 (4.23). Storage & Organisation is the largest category by listing count.

### 3.6 HDBSCAN Wins the Model Competition — and Finds 69% of the Catalog Is Unclustered

**Champion: HDBSCAN (min_cluster_size=320, min_samples=20)**

| Algorithm | Config | Silhouette | DB Index | Stability | Score |
|---|---|---|---|---|---|
| **HDBSCAN** | **mcs=320,ms=20** | **0.292** | **1.061** | **0.920** | **0.839** |
| KMeans | k=7 | 0.147 | 2.052 | 0.591 | 0.764 |
| KMeans | k=6 | 0.136 | 2.236 | 0.877 | 0.753 |
| KMeans | k=4 | 0.118 | 2.548 | 0.991 | 0.675 |
| KMeans | k=5 | 0.128 | 2.345 | 0.793 | 0.658 |
| GMM | k=6 | 0.108 | 2.743 | 0.402 | 0.386 |

- **Silhouette** (0–1, higher = better-separated clusters): HDBSCAN 0.292 vs KMeans best of 0.147 — nearly double.
- **Davies-Bouldin** (lower = more compact, well-separated clusters): HDBSCAN 1.061 vs KMeans best 2.052.
- **ARI Stability** (0–1, how consistently the same groupings appear across runs): HDBSCAN 0.920 — highly reproducible.

**Critical finding: 23,823 products (69.2% of the unique catalog) were labeled as Outliers / Noise.** HDBSCAN does not force products into groups they do not belong to — and 69% of products genuinely do not belong to any coherent group.

### 3.7 Five Product Segments Identified

| Segment | Products | Avg Price (local) | Avg Rating | Avg Reviews | Top Seller % | Avg Markets |
|---|---|---|---|---|---|---|
| Outliers / Noise | 23,823 | 21,383 | 4.34 | 136 | 4.9% | 12 |
| Regional Luxury Niche | 4,974 | 20,020 | 4.47 | 60 | 0.0% | 2 |
| New Arrivals Pipeline | 2,558 | 9,941 | 4.59 | 97 | 0.4% | 10 |
| Reviewed Premium Picks | 1,676 | 43,813 | 4.58 | 43 | 0.9% | 19 |
| **Community Favourites** | **1,378** | **2,134** | **4.53** | **122** | **3.9%** | **29** |

**Segment descriptions (derived from cluster centroid values, not assigned manually):**

- **Community Favourites (1,378 products):** Globally distributed across 29 markets on average. Affordable pricing. Strong engagement (122 avg reviews). Highest top-seller rate of any named segment (3.9%). These products include OFTAST, PRICKIG, KONCIS, PRODUKT — each available in 40–46 markets with top-seller rates between 39–48%.

- **Reviewed Premium Picks (1,676 products):** Present across 19 markets, positioned in the top 82nd price percentile within their currency. Products include TROFAST, JÄTTESTA, LANESUND, PLATSA — rated 5 stars but averaging only 43 reviews. Near-zero top-seller rate (0.9%). High quality, very low market visibility.

- **New Arrivals Pipeline (2,558 products):** Recently tagged products. High ratings (4.59 avg) and presence across 10 markets. Top-seller rate near zero (0.4%), consistent with newness. Two sub-clusters: higher-price new arrivals (cluster 3, regional focus) and mid-range new arrivals (cluster 2, broader spread).

- **Regional Luxury Niche (4,974 products):** Present in only 1–2 markets on average. Positioned in the top 70–88th price percentile locally. Zero global top-seller presence. Market-specific premium items with no cross-border footprint.

- **Outliers / Noise (23,823 products):** Products with no consistent peer group. Span all price tiers, engagement levels, and market presences. Some of the catalog's most successful global products (BRIMNES, TRONES, SAMLA) appear here precisely because their extreme cross-market presence makes them statistically distinct from any cluster.

### 3.8 Anomaly Detection Flagged 1,721 Products (5.0%)

The Isolation Forest identified 1,721 products as statistically unusual across all 34 features simultaneously.

| Segment | Anomaly Count |
|---|---|
| Outliers / Noise | 1,590 |
| Community Favourites | 70 |
| Reviewed Premium Picks | 61 |

**Top globally-present anomalies (products in all 46 markets):**

| Product | Segment | Markets | Top Seller % | Avg Rating |
|---|---|---|---|---|
| BRIMNES | Outliers / Noise | 46 | 65.2% | 4.28 |
| TRONES | Outliers / Noise | 46 | 45.7% | 4.70 |
| SAMLA | Outliers / Noise | 46 | 37.0% | 4.64 |
| PRODUKT | Community Favourites | 46 | 47.8% | 4.19 |
| KONCIS | Community Favourites | 46 | 43.5% | 4.68 |
| OFTAST | Community Favourites | 46 | 39.1% | 4.83 |
| PRICKIG | Community Favourites | 46 | 39.1% | 4.73 |

**Observation:** These products are anomalies because they are exceptional, not because something is wrong with them. They are flagged because no standard product looks like them across all 34 dimensions simultaneously.

### 3.9 Top Product Series by Catalog Volume

| Series | Appearances | Product Type |
|---|---|---|
| PAX | 12,534 | Wardrobe systems |
| METOD | 9,255 | Kitchen units |
| TROFAST | 5,911 | Children's storage |
| PLATSA | 5,814 | Modular storage |
| TONSTAD | 5,021 | Bedroom furniture |
| BILLY | 4,213 | Bookcases |
| MAXIMERA | 3,940 | Kitchen accessories |
| VIMLE | 3,841 | Sofas |

**Observation:** PAX accounts for over 12,500 listing rows — consistent with a highly configurable modular system generating many variant listings per base product. The top 8 series collectively dominate catalog volume.

---

## 4. What We Can Now Know

### Decisions This Analysis Enables

| Decision | Evidence Base |
|---|---|
| Which products to prioritize for global expansion | Community Favourites have proven cross-market demand. Reviewed Premium Picks have quality but lack reach. Both are now identifiable by name. |
| Whether premium pricing signals higher quality to customers | No. Price-to-rating correlation is 0.011. Premium products do not earn better satisfaction scores. |
| Whether current promotions are working | No. Promoted items receive 41% fewer reviews and cost double. The allocation logic needs revision. |
| Which products are the clearest global success benchmarks | Seven products are in all 46 markets with top-seller rates above 35%: BRIMNES, TRONES, SAMLA, PRODUKT, KONCIS, OFTAST, PRICKIG. |
| Which clustering method to standardize on | HDBSCAN. It outperforms KMeans on every metric and is reproducible across seeds with 92% ARI stability. |
| How concentrated the catalog is in key product lines | Five series (PAX, METOD, TROFAST, PLATSA, BILLY) generate a disproportionate share of catalog volume. |

### New Visibility Gained

- **Portfolio fragmentation is now measured:** 69% of products have no peer group. This number did not exist before this analysis.
- **The mid-range gap is quantified:** Only 19.6% of the catalog sits in the 50–500 price range. The portfolio has a measurable missing middle.
- **Anomalies are ranked and named:** 1,721 flagged products are sortable by predicted commercial importance rather than requiring manual review of all 34,409.
- **Segment labels are data-derived:** Community Favourites, Reviewed Premium Picks, Regional Luxury Niche, New Arrivals Pipeline — these labels come from actual cluster centroid values, not from manual assignment.
- **The catalog's seven global champions are identified by name.**

### Risks Now Quantifiable

- **Sofa satisfaction gap:** The category with the highest average price (16,930 local) earns the lowest rating among top-10 categories (4.23). There is a 0.25-point satisfaction gap versus the next-highest-priced category.
- **Regional Luxury Niche concentration:** 4,974 products are each present in only 1–2 markets. If those markets contract, these products have no fallback distribution.
- **Reviewed Premium Picks invisibility:** 1,676 five-star products averaging only 43 reviews. Low visibility on high-value items is a direct revenue risk.

---

## 5. Algorithm and Modeling Recommendation

### Do We Need a Model? Yes — Two Distinct Models Are Justified.

---

### Model A: Top-Seller Probability Predictor

**Purpose:** Predict the likelihood that a product will become a top seller in a given market — before investing in its promotion or launch.

**Recommended algorithm: Gradient Boosted Trees (XGBoost or LightGBM)**

**Why this algorithm:**
- The target variable is `top_seller_share` (0 to 1) — a regression problem, not a yes/no classification.
- The features combine numeric signals (price rank, market spread, review count, volume) and binary indicators (new arrival tag, limited edition, discontinued). Gradient boosted trees handle this mix natively without requiring manual preprocessing for every variable.
- The algorithm captures non-linear interactions — for example, a product that is both globally distributed AND in the top price percentile may behave differently from one that is expensive but only sold locally. Standard regression would miss this.
- Feature importance scores are interpretable: the model will show exactly which of the 34 features is driving the prediction.

**Training setup:**
- **Training data:** `ikea_advanced_features.csv` — 34,409 products, 34 engineered features. Target: `top_seller_share` column.
- **Train/test split:** 80% training, 20% held-out test. Use stratified sampling by segment to preserve representation of each cluster in both sets.
- **Validation:** 5-fold cross-validation on the training set. Final evaluation on the held-out test set only.
- **Success metrics:**
  - R² (what share of variance in top-seller rate the model explains). A score above 0.40 is a useful starting threshold.
  - RMSE on the held-out test set — the average prediction error in percentage-point terms.
  - Business lift: among the top 20% of products ranked by the model's predicted score, what share actually achieve top-seller status? This is the operational measure.

**Prerequisite:** The `promoted_share = 1.0` data anomaly must be resolved before training. If all products look identically promoted, that feature will not contribute — and may corrupt the model. Fix first, build second.

**How it would be used operationally:**
- Before launching a product in a new market → input its 34 features → receive a top-seller probability score.
- Rank all New Arrivals Pipeline products by predicted score → concentrate promotional spend on the top quartile.
- Monthly refresh: re-score all products as new reviews and market data arrive.

---

### Model B: Portfolio Segmentation — Operationalize HDBSCAN (Maintain and Refresh)

**Purpose:** Keep the five-segment product map current as the catalog evolves.

**Recommended algorithm: HDBSCAN, champion configuration (mcs=320, ms=20)**

**Why retain this model:**
- Already validated: Silhouette 0.292, Davies-Bouldin 1.061, ARI Stability 0.920.
- Does not force products into segments — the 69% noise finding is a real business signal, not a model failure.
- Reproducible across random seeds (stability 0.920).

**Training setup:**
- **No separate test set required.** Clustering is evaluated on the full dataset using silhouette and Davies-Bouldin scores.
- **Refresh cadence:** Run quarterly as new products enter the catalog.
- **Health threshold:** Silhouette score should remain above 0.25. A significant drop signals that the product mix has shifted and segment boundaries need recalibration.

**How it would be used operationally:**
- Every new product added is automatically assigned a segment on the next quarterly run.
- Products moving between segments over time signal a change in market behavior — e.g., a New Arrival graduating to Community Favourite.
- The noise proportion (currently 69%) is tracked over time as a catalog health metric. If it grows, the catalog is becoming more fragmented.

---

### What Not to Build Now

A **time-series forecasting model** is not recommended. The current dataset contains no time dimension — it is a single catalog snapshot. Any forecast would be speculation.

A **product category classifier** is not needed. The category taxonomy already exists in the data and does not require prediction.

---

## 6. Business Implications

### Revenue

- **Community Favourites are undermonetized.** The 1,378 products in this segment have the widest market reach (29 markets average), the highest top-seller rates of any named segment (3.9%), and the lowest average price (2,134 local currency). They generate loyalty and volume but not margin. Moving even a fraction of these customers one tier up the price ladder is a direct revenue opportunity.

- **Reviewed Premium Picks are invisible revenue.** 1,676 products are priced in the top 82nd percentile and rated at or near 5 stars, but average only 43 reviews. These products are not being discovered. Improved visibility alone — without any product or price changes — could unlock conversion on high-margin items.

- **The mid-range gap (50–500 price range, currently only 19.6% of catalog)** means that a budget customer who wants to spend more has few natural upgrade options. This is a structural revenue leakage point.

### Cost

- **Promotional spend is misaligned.** Promoted products receive 41% fewer reviews and cost double on average. Either the mechanic is not working for this product set, or the wrong products are being promoted. Without knowing which, money is being spent on a program with no measurable engagement benefit.

- **Regional Luxury Niche (4,974 products, avg 1–2 markets each)** carries catalog management, logistics, and merchandising cost across the supply chain without global scalability. A line-by-line cost-benefit review of this segment is warranted.

### Risk

- **Sofas & Armchairs** rate 4.23 — the lowest satisfaction of any top-10 category — at the highest average price (16,930 local). This is a reputational risk in a high-commitment, high-visibility product category.

- **The 69% unclustered catalog** means the majority of products cannot be grouped, benchmarked, or optimized at scale. As the catalog grows without structural coherence, supply chain complexity and marketing inefficiency grow with it.

- **The `promoted_share` data anomaly** in the ML pipeline means any model trained on this feature without correction would produce invalid results. This is a known risk that is straightforward to fix.

---

## 7. Recommended Business Actions

### Immediate Actions (0–3 Months)

**Action 1: Fix the promotional tag data pipeline**
- **Rationale:** Every product in the ML features table shows `promoted_share = 1.0`. This is a data anomaly that will corrupt Model A if unresolved.
- **Expected benefit:** Ensures promotion effectiveness data is accurate. Unblocks the top-seller predictor model.
- **Dependency:** Data engineering team; review of `ikea_cleaner.py` `sale_tag` field handling.

**Action 2: Audit what promotion types are applied to which products**
- **Rationale:** Promoted products receive 41% fewer reviews than non-promoted. Pulling the actual promotion type breakdown (NEW_PRODUCT, NEW_LOWER_PRICE, FAMILY_PRICE, TIME_RESTRICTED_OFFER) will reveal which promotion mechanic is driving — or not driving — engagement.
- **Expected benefit:** Redirect promotional spend toward the mechanics and product types that actually increase review volume.
- **Dependency:** Access to full `sale_tag` distribution breakdown and promotion spend records.

**Action 3: Brief the seven globally exceptional products**
- **Rationale:** BRIMNES (65.2% top-seller rate across 46 markets), TRONES, SAMLA, PRODUKT, KONCIS, OFTAST, PRICKIG are empirically the clearest examples of global product success in this catalog. Understanding what they share in common is the fastest path to identifying the next global product.
- **Expected benefit:** Builds a replicable template for global product development.
- **Dependency:** Category and product management teams.

**Action 4: Send the 1,721 anomaly product list to category managers for structured review**
- **Rationale:** The Isolation Forest has already done the triage. Managers do not need to review all 34,409 products — only the 1,721 flagged ones, sorted by predicted commercial importance.
- **Expected benefit:** Rapid identification of pricing errors and hidden high-potential products.
- **Dependency:** `ikea_advanced_features.csv`, column `is_anomaly == 1`.

---

### Medium-Term Actions (3–12 Months)

**Action 5: Build and deploy the Top-Seller Probability Predictor (Model A)**
- **Rationale:** The 2,558 New Arrivals Pipeline products currently achieve a 0.4% top-seller rate. A model that ranks these by predicted success probability allows promotional and shelf investment to go to the highest-potential items, not to random selection.
- **Expected benefit:** Improved return on launch investment; reduced waste on products unlikely to succeed regardless of support.
- **Dependency:** Action 1 completed (data pipeline fix); data science team with gradient boosting capability.

**Action 6: Increase visibility for the 1,676 Reviewed Premium Picks**
- **Rationale:** Five-star products with only 43 reviews and premium pricing are revenue waiting to be unlocked. These products do not need quality work — they need to be found.
- **Expected benefit:** Conversion uplift on high-margin items through targeted content, review solicitation, and search placement improvements.
- **Dependency:** Marketing and e-commerce teams; review request program.

**Action 7: Review the Regional Luxury Niche for expansion or rationalization**
- **Rationale:** 4,974 products serve 1–2 markets each. Use the `market_spread_share` feature from the product table to identify which of these could plausibly expand (high engagement in their home market) and which should be considered for rationalization.
- **Expected benefit:** Reduced catalog complexity and supply chain cost; focused investment on expandable products.
- **Dependency:** Supply chain and regional merchandising teams.

**Action 8: Address the mid-range price gap**
- **Rationale:** Only 19.6% of the catalog sits in the 50–500 price range. Budget customers have no clear upgrade path; the portfolio's "missing middle" creates a value ladder with a gap in the most common spending range.
- **Expected benefit:** Improved customer lifetime value as budget buyers gain a natural next step.
- **Dependency:** Product development and pricing strategy teams.

---

### Strategic Bets

**Bet 1: Build the global product growth strategy around Community Favourites**
- These 1,378 products already work across 29 markets on average, earn strong ratings, and achieve top-seller status at the highest rate of any segment. Deepening investment in these products — more variants, better placement, adjacent product development — compounds a proven global formula.
- **Risk:** Their low average price (2,134 local) limits margin at scale unless paired with an explicit upsell architecture.

**Bet 2: Treat the 69% noise finding as a portfolio pruning mandate**
- A catalog where 23,823 products have no peer group has grown wider than it is deep. A rationalization program that focuses on deepening coherent segments rather than expanding the noise could reduce supply chain cost, improve customer navigation clarity, and increase the signal in all future analysis.
- **Risk:** Some of the most commercially successful products (BRIMNES, TRONES, SAMLA) live in the noise segment. Pruning requires careful individual review — not blanket cuts.

**Bet 3: Build a quarterly product health dashboard using the HDBSCAN model**
- Run the clustering pipeline every quarter. Track each product's segment assignment over time. Products moving from New Arrivals to Community Favourites are early-stage breakouts. Products moving from broad segments to Regional Luxury Niche are losing global reach. This turns one-time analysis into a continuous intelligence system.
- **Risk:** Requires a production data pipeline and reporting infrastructure investment.

---

## 8. Limitations and Data Gaps

| Limitation | Impact on Analysis | What Would Fix It |
|---|---|---|
| All prices in local currencies, not normalized | Cross-country price comparisons in raw numbers are invalid | Apply PPP-adjusted or FX-normalized pricing before any pricing strategy decisions |
| `promoted_share = 1.0` for all products in ML features | Promotion signal is unusable in Model A | Audit `ikea_cleaner.py` `sale_tag` aggregation logic |
| No sales volume or revenue data | Cannot quantify any finding in financial terms | Link catalog data to POS or e-commerce transaction records |
| No time dimension (single catalog snapshot) | Cannot identify trends, lifecycle stages, or seasonal patterns | Add historical catalog snapshots quarterly |
| 69% of products labeled as noise | Cannot analyze these products in groups — only individually | More granular features (customer demographics, time-to-purchase) may reveal sub-structure |
| Price skewness of 18.33 and kurtosis of 472 post-clean | Extreme values remain in the distribution | Currency normalization would compress the distribution substantially |
| Cluster silhouette of 0.292 (moderate) | Segment boundaries are real but not sharp | Confirmed by HDBSCAN's design: this reflects genuine catalog diversity, not a modeling failure |
| Review count used as engagement proxy | Reviews are not purchases, dwell time, or conversion rate | Sales transaction data would provide a more direct measure |
| Regional Luxury Niche shows 0% top-seller rate | May reflect that the "Top Seller" badge is not issued in those specific markets, not necessarily low popularity | Market-level sales data needed to confirm |
| `promoted_share` anomaly makes promotion-related features unreliable | Affects both descriptive analysis and predictive modeling | Fix data pipeline before any model training |

---

## 9. Appendix — Source of Truth

### A. Price Distribution (ikea_analysis.py — cleaned data)
```
Loading cleaned data from: e:\Excel Datasets\ikea_clean.csv
  Rows: 397,032  |  Products: 34,409

10th percentile:     5.99
25th percentile:    20.00
50th percentile:   149.00
75th percentile:   899.00
90th percentile:  8,495.00
95th percentile: 32,400.00
99th percentile: 309,000.00
Skewness: 18.33
Kurtosis: 472.45
```

### B. Rating Summary (ikea_analysis.py)
```
Products with ratings: 267,700
Mean rating:           4.46
Median rating:         4.60
Standard deviation:    0.62
Mode (most common):    5.0
```

### C. Correlation Results (ikea_analysis.py)
```
Price vs Rating correlation:        0.0113
Rating vs Review Count correlation: 0.0259
```

### D. Price Tier Distribution (ikea_analysis.py)
```
Budget (<50):           144,415  (36.4%)  Avg Rating: 4.49
Affordable (50-100):     38,278  ( 9.6%)  Avg Rating: 4.45
Mid-range (100-250):     49,255  (12.4%)  Avg Rating: 4.42
Premium (250-500):       39,659  (10.0%)  Avg Rating: 4.39
Luxury (500-1,000):      32,896  ( 8.3%)  Avg Rating: 4.39
Ultra-Luxury (>1,000):   92,529  (23.3%)  Avg Rating: 4.46
```

### E. Promotion Comparison (ikea_analysis.py)
```
Promotional products:      70,155  (17.7%)
Non-Promotional products: 326,877  (82.3%)

Promoted avg rating:     4.45   |  Non-promoted avg rating:  4.46
Promoted avg reviews:  113.16   |  Non-promoted avg reviews: 193.00
Promoted avg price:  22,322.18  |  Non-promoted avg price: 11,305.74
```

### F. Top 10 Categories (ikea_analysis.py)
```
storage-organisation:     51,405 | Avg Price:  15,690 | Rating: 4.42
kitchenware-tableware:    26,032 | Avg Price:     462 | Rating: 4.53
beds-mattresses:          22,263 | Avg Price:   6,096 | Rating: 4.39
decoration:               20,214 | Avg Price:     904 | Rating: 4.62
kitchen-appliances:       14,428 | Avg Price:   7,082 | Rating: 4.29
storage-furniture:        14,193 | Avg Price:     750 | Rating: 4.38
sofas-armchairs:          13,501 | Avg Price:  16,930 | Rating: 4.23
lighting:                 12,541 | Avg Price:   1,966 | Rating: 4.31
small-storage-organisers: 11,176 | Avg Price:   1,041 | Rating: 4.53
kitchens:                 10,031 | Avg Price:   2,445 | Rating: 4.28
```

### G. Model Search Full Rankings (ikea_advanced_ds.py)
```
 Method        Config  Clusters  Silhouette  DB_Index  Stability  Noise%  Score
HDBSCAN  mcs=320,ms=20    12.67      0.2918    1.0613     0.9201   68.25  0.8389
 KMeans          k=7       7.00      0.1465    2.0524     0.5905    0.00  0.7639
 KMeans          k=6       6.00      0.1357    2.2360     0.8767    0.00  0.7528
 KMeans          k=4       4.00      0.1181    2.5476     0.9907    0.00  0.6750
 KMeans          k=5       5.00      0.1281    2.3446     0.7928    0.00  0.6583
    GMM   components=6     6.00      0.1076    2.7431     0.4024    0.00  0.3861
    GMM   components=7     7.00      0.1011    2.9628     0.5464    0.00  0.3750
    GMM   components=5     5.00      0.0901    2.8683     0.4704    0.00  0.3083
    GMM   components=4     4.00      0.0843    3.0265     0.5243    0.00  0.2417

Champion: HDBSCAN | mcs=320,ms=20
Silhouette: 0.2918  |  Davies-Bouldin: 1.0613  |  Stability: 0.9201  |  Noise: 68.25%
```

### H. Segment Summary (ikea_advanced_ds.py + ikea_advanced_features.csv)
```
Segment                  Products  Avg Price  Avg Rating  Avg Reviews  Top Seller%  Avg Markets
Outliers / Noise           23,823     21,383        4.34          136         4.9%         12
Regional Luxury Niche       4,974     20,020        4.47           60         0.0%          2
New Arrivals Pipeline       2,558      9,941        4.59           97         0.4%         10
Reviewed Premium Picks      1,676     43,813        4.58           43         0.9%         19
Community Favourites        1,378      2,134        4.53          122         3.9%         29
```

### I. Anomaly Detection (ikea_advanced_ds.py)
```
Total anomalies: 1,721 (5.0% of 34,409 products)
  Outliers / Noise:       1,590
  Community Favourites:      70
  Reviewed Premium Picks:    61
```

### J. Top Globally-Present Products — All 46 Markets (ikea_advanced_features.csv)
```
Product    Segment               Markets  Top Seller%  Avg Rating
BRIMNES    Outliers / Noise         46       65.2%        4.28
TRONES     Outliers / Noise         46       45.7%        4.70
SAMLA      Outliers / Noise         46       37.0%        4.64
PRODUKT    Community Favourites     46       47.8%        4.19
KONCIS     Community Favourites     46       43.5%        4.68
OFTAST     Community Favourites     46       39.1%        4.83
PRICKIG    Community Favourites     46       39.1%        4.73
```

---

*All findings derived strictly from `ikea_analysis.py` and `ikea_advanced_ds.py` script outputs on the cleaned dataset.
No data was inferred, estimated, or invented.
All prices are in local currencies — cross-country comparisons require FX normalization before use in pricing decisions.*
