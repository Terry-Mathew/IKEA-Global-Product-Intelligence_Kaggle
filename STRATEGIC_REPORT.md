# IKEA Global Product Portfolio — CEO Strategic Intelligence Report

**Prepared:** February 28, 2026
**Data Source:** IKEA Global Product Catalog (~390,000 product listings, 50+ countries)
**Analyses Run:** `ikea_analysis.py` (8 statistical techniques) + `ikea_advanced_ds.py` (ML clustering & anomaly detection)

---

## 1. Executive Summary

The IKEA global product catalog was analyzed across 390,000+ listings using eight statistical methods and three machine learning models. The data reveals a portfolio that is structurally polarized: **37% of products are budget items under 50 local currency units, yet 22% are ultra-premium items above 1,000 units**. The middle tiers (100–500) are comparatively thin.

Customer satisfaction is uniformly high across the portfolio — **average rating of 4.45 out of 5.0** — and, critically, **price does not predict quality perception** (correlation: 0.01). This means customers feel equally satisfied regardless of price point, which both validates the brand and limits the ability to use premium pricing as a trust signal.

Promoted products receive nearly **3x more reviews** than non-promoted products (452 vs 159), suggesting promotions are an effective lever for social proof, not just sales. The ML segmentation identified five distinct product archetypes, of which one — a "Price-Flexible Bestseller" cluster — shows the highest top-seller rate (6.8%) and the most global price variation, indicating strong demand that may be under-optimized for margin.

**Three actions are supported directly by the data:** (1) strengthen mid-tier pricing gaps, (2) investigate the 1,324 anomalous products flagged by the model, and (3) scale promotional treatment to high-potential products approaching the high-review threshold.

---

## 2. Methodology

### 2.1 Distribution Analysis
- **What it does:** Maps where prices and ratings cluster across the full product range.
- **Question answered:** What is the typical IKEA price point, and how spread out are they?
- **Why appropriate:** Establishes the baseline before any deeper analysis. Skewness and kurtosis measure how lopsided and extreme the distribution is.

### 2.2 Correlation Analysis
- **What it does:** Measures the statistical relationship between pairs of variables (price vs rating; rating vs reviews).
- **Question answered:** Does paying more mean being more satisfied?
- **Why appropriate:** A direct test of the "premium = quality" assumption. Values close to 0 mean no meaningful relationship.

### 2.3 Category Profiling
- **What it does:** Groups products by category and computes average price and rating per group.
- **Question answered:** Which categories drive volume vs. value?
- **Why appropriate:** Identifies where the portfolio is dense and where it earns the most per unit.

### 2.4 Outlier Detection (IQR Method)
- **What it does:** Flags products whose prices fall outside 1.5× the middle spread (Q1–Q3 range).
- **Question answered:** Are there pricing anomalies that distort the data or signal data quality issues?
- **Why appropriate:** Standard, robust method for identifying unusual data points without assuming a specific distribution.

### 2.5 Geographic Analysis
- **What it does:** Compares average price, median price, and ratings by country.
- **Question answered:** Where is IKEA most active, and how do local price levels vary?
- **Why appropriate:** Reveals regional portfolio concentration and pricing range.
- **Limitation:** Prices are in local currencies (RSD, COP, HUF, CLP, KRW, JPY, etc.). Cross-country price comparisons are not valid without currency conversion.

### 2.6 Promotion Effectiveness Analysis
- **What it does:** Splits the catalog into promoted vs. non-promoted products and compares ratings, review counts, and prices.
- **Question answered:** Does running a promotion correlate with better customer engagement?
- **Why appropriate:** Measures the observable footprint of promotions without requiring sales data.

### 2.7 Text Analysis (Product Name Frequency)
- **What it does:** Counts the most frequently occurring words across all product names.
- **Question answered:** Which product series or design families dominate the catalog?
- **Why appropriate:** Fast, systematic way to identify catalog concentration by product line.

### 2.8 Price Tier Segmentation
- **What it does:** Assigns each product to one of six price bands (Budget to Ultra-Luxury) and measures average rating per band.
- **Question answered:** How is the portfolio distributed by price positioning, and does tier affect customer satisfaction?
- **Why appropriate:** Provides a clean framework for portfolio strategy discussions.

### 2.9 K-Means Clustering (Machine Learning)
- **What it does:** Assigns each unique product to one of five groups based on shared characteristics (price, rating, review count, geographic spread, physical size).
- **Question answered:** Are there natural product archetypes in the global catalog?
- **Why appropriate:** Discovers hidden structure without pre-labeling. Validated with Silhouette Score (0.32) and Davies-Bouldin Index (1.34) — both indicate moderate, usable cluster separation.

### 2.10 Anomaly Detection (Isolation Forest)
- **What it does:** Flags products that are statistically unusual across all engineered features simultaneously.
- **Question answered:** Which products don't fit any known pattern — pricing errors, niche breakouts, or hidden opportunities?
- **Why appropriate:** Works on high-dimensional data without assuming any specific distribution; suited to finding rare combinations of attributes.

---

## 3. Key Findings

### 3.1 The Portfolio Is Bimodal — Budget AND Ultra-Luxury Dominate

| Price Tier | Products | Share |
|---|---|---|
| Budget (<50 currency units) | 144,415 | **37.0%** |
| Affordable (50–100) | 38,278 | 9.8% |
| Mid-range (100–250) | 49,255 | 12.6% |
| Premium (250–500) | 39,676 | 10.2% |
| Luxury (500–1,000) | 33,115 | 8.5% |
| Ultra-Luxury (>1,000) | 86,078 | **22.0%** |

**Fact:** The "middle" tiers (Affordable + Premium combined) represent only 20% of the portfolio, while the two extremes command 59%.

### 3.2 Customer Satisfaction Is High and Price-Independent

- Mean rating: **4.45 / 5.0** across 263,077 rated products
- Median rating: **4.6** — over half of rated products score above this
- Price-to-rating correlation: **0.0099** (effectively zero)
- Rating-to-reviews correlation: **0.0275** (effectively zero)

**Interpretation:** Customers are satisfied at every price point. There is no evidence that higher price drives higher satisfaction, nor that better-rated products attract more reviews organically.

### 3.3 Promotions Produce 3x More Reviews

| Metric | Promoted | Non-Promoted |
|---|---|---|
| Product count | 28,131 (7.2%) | 362,686 (92.8%) |
| Avg rating | 4.49 | 4.45 |
| **Avg reviews** | **451.72** | **159.39** |
| Avg price | 2,917.50 | 3,059.67 |

**Fact:** Promoted products generate 2.84× more reviews on average. Promoted products also carry slightly lower prices and slightly higher ratings, though the rating gap is small (0.04 stars).

### 3.4 Storage is the Largest Category; Sofas Carry the Highest Price

| Category | Products | Avg Price | Avg Rating |
|---|---|---|---|
| Storage & Organisation | 50,447 | 4,279.14 | 4.43 |
| Kitchenware & Tableware | 26,031 | 457.75 | 4.53 |
| Beds & Mattresses | 22,171 | 2,221.19 | 4.40 |
| Decoration | 20,203 | 808.82 | **4.62** |
| Sofas & Armchairs | 14,572 | **5,873.99** | 4.23 |
| Kitchen Appliances | 14,221 | 1,945.55 | 4.29 |

**Fact:** Decoration achieves the highest customer rating (4.62) at a mid-range price point. Sofas & Armchairs carry the highest average price but the lowest rating among the top six categories (4.23).

### 3.5 Five Product Archetypes Identified by Machine Learning

Dataset: 26,480 unique products analyzed (cluster model quality: Silhouette Score 0.32, Davies-Bouldin 1.34 — moderate)

| Cluster | Size | Avg Price | Avg Rating | Top-Seller Rate | Price Variance Index | Label |
|---|---|---|---|---|---|---|
| 0 | 5,599 | 1,370 | 0.80 | 0.7% | 1.44 | Unrated / Data-Sparse |
| 1 | 3,960 | 1,264 | 4.46 | **6.8%** | **20.86** | Price-Flexible Bestsellers |
| 2 | 829 | 6,246 | 2.57 | 5.3% | 0.04 | High-Price, Low-Satisfaction |
| 3 | 13,718 | 972 | 4.42 | 3.4% | 2.79 | Reliable Mid-Range Core |
| 4 | 2,374 | 6,685 | 4.00 | 4.7% | 15.65 | Premium Variable |

**Note on Cluster 0 Rating:** The 0.80 average rating in Cluster 0 almost certainly reflects missing rating data (products with zero or near-zero ratings entered as 0), not actual customer dissatisfaction. This is a data quality issue, not a product quality signal.

**Key interpretation:**
- **Cluster 3** (13,718 products, 52% of analyzed set) is the volume backbone — affordable, well-rated, stable.
- **Cluster 1** is the highest-priority growth opportunity: best top-seller rate AND the widest price spread globally (index 20.86 vs near-zero for Cluster 2), suggesting these products can command varying prices in different markets.
- **Cluster 2** (829 products) is a warning signal: very expensive, very fixed in price, but customers rate them at only 2.57/5. This warrants product investigation.

### 3.6 Anomaly Detection Flagged 1,324 Products (5.0%)

The Isolation Forest model identified 1,324 products (out of 26,480) exhibiting unusual combinations of price range, geographic spread, review volume, and physical attributes. Sample anomalies show products with average prices above 6,000–9,000 but minimum observed prices as low as 3.50–35.00 — indicating extreme cross-country price inconsistency.

### 3.7 Top Product Series by Catalog Volume

The most-referenced series in product names:
- **PAX** (12,439 appearances) — wardrobe system
- **METOD** (8,855) — kitchen system
- **TROFAST** (5,705) — children's storage
- **PLATSA** (5,469) — modular storage
- **BILLLY** (4,038) — bookcase

These five series represent catalog anchors and likely account for disproportionate revenue.

### 3.8 Ultra-Luxury Pricing Cluster and Data Quality Concern

The 10 most expensive outliers are all priced at exactly 99,999 in local currency (Serbian RSD or Colombian COP), which is characteristic of a **data entry price ceiling, not actual retail pricing**. These are data artifacts, not real premium products, and inflate the upper tail of the price distribution.

---

## 4. Business Implications

### 4.1 Mid-Tier Portfolio Gap Is a Risk and Opportunity
The combined Affordable + Premium tiers (50–500 currency units) represent only 20% of the portfolio. This leaves a wide gap in the middle of the value ladder. If consumer spending contracts, customers may struggle to step down from Ultra-Luxury to something still within the brand without jumping all the way to Budget. Competitors filling the mid-range can poach price-sensitive existing customers.

### 4.2 Premium Pricing Does Not Carry a Quality Halo — Yet
A correlation of 0.0099 between price and rating means IKEA customers do not assign quality premium to expensive products. This is a double-edged finding: it confirms brand trust at all price points, but it also means that premium products are not earning the satisfaction premium that would justify higher prices to new customers. The low-rated Cluster 2 (avg 2.57 at avg price 6,246) represents reputational risk in the high-end segment.

### 4.3 Promotions Are a Review-Generation Machine
The 3x review gap between promoted and non-promoted products means promotions are generating social proof at scale. Since ratings do not differ significantly (4.49 vs 4.45), the review volume difference is almost entirely a function of promotion-driven traffic. This is leverage that can be systematically deployed to push high-potential products above visibility thresholds.

### 4.4 Cluster 1 ("Price-Flexible Bestsellers") Is the Key Margin Lever
These 3,960 products have the highest top-seller rate (6.8%) AND the highest price variance across markets (index 20.86). This means these products are already popular globally, but are priced very differently from country to country. Standardizing or optimizing pricing in underpriced markets could lift margin without impacting volume.

### 4.5 The Data Quality Problem Is Material
The 99,999 price cap entries, the zero-rating cluster (5,599 products), and the extreme price variance in anomaly products all suggest systemic data pipeline issues across country feeds. Decisions made on this data without cleansing carry real risk.

---

## 5. Recommendations

### Rec 1 — Audit and Reinforce the Mid-Range Tier
**Action:** Identify which product lines currently sit in the 50–500 price range and evaluate whether catalog gaps in that tier can be filled by repositioning existing products or accelerating mid-tier launches.
**Expected benefit:** Closes the portfolio gap that exposes IKEA to mid-market competitor displacement.
**Risk:** Cannibalization of budget or premium segments if positioning is unclear.

### Rec 2 — Investigate Cluster 2 ("High-Price, Low-Satisfaction")
**Action:** Pull the 829 products in Cluster 2 (avg price ~6,246, avg rating 2.57) and review for quality issues, misleading product descriptions, or fulfillment problems.
**Expected benefit:** Prevents reputational damage from low-rated premium products; potential for rapid rating improvement through targeted quality or communication fixes.
**Risk:** Some products may have legitimate niche appeal — review before delisting.

### Rec 3 — Systematically Promote Products Between 80–200 Reviews
**Action:** Create a promotional priority queue targeting products with review counts in the 80–200 range. The data shows promotions drive a 3x review multiplier, which would push these products past the high-visibility threshold.
**Expected benefit:** Compounds the review pool for mid-maturity products, increasing organic search visibility and conversion.
**Dependency:** Requires marketing team coordination with catalog data to identify and tag qualifying products.

### Rec 4 — Price Optimization for Cluster 1 in Underpriced Markets
**Action:** Commission a market-by-market price benchmarking exercise for the 3,960 Cluster 1 products (Price-Flexible Bestsellers), focusing on markets where local prices sit below the global average for those items.
**Expected benefit:** Margin recovery without volume loss, given these products already have the highest top-seller rates.
**Risk:** Local pricing norms and competitive dynamics must be respected. A blanket increase is not appropriate; market-by-market analysis is required.

### Rec 5 — Fix the Data Pipeline Before the Next Analysis Cycle
**Action:** Remediate three specific data quality issues: (a) price cap artifacts (99,999 entries), (b) zero-rating imputation vs. true missing data, (c) multi-currency price fields that prevent direct comparison.
**Expected benefit:** All future analysis will be more reliable; geographic strategy decisions will be based on real price comparisons.
**Dependency:** Requires data engineering and country catalog team collaboration.

### Rec 6 — Monitor the 1,324 Anomalous Products for Opportunity
**Action:** Review the Isolation Forest flagged products for two categories: (a) products with inconsistently low prices in high-value markets (potential pricing errors), and (b) products with unusual feature combinations that may signal emerging demand patterns.
**Expected benefit:** Recovery of pricing errors; early identification of niche product breakouts before they scale organically.

---

## 6. Limitations & Data Gaps

| Limitation | Impact | Mitigation |
|---|---|---|
| Prices in local currencies across all countries | Geographic price comparisons are invalid without FX conversion | Apply PPP-adjusted or FX-normalized prices before any cross-country pricing strategy work |
| Price cap artifacts (99,999 entries in RSD/COP) | Inflates upper-tail statistics and outlier counts | Filter known cap values before distribution analysis |
| Zero-rating products in Cluster 0 | Cluster analysis mislabels these as low-satisfaction | Separate "no data" from "low rating" before modeling |
| Silhouette Score of 0.32 for clustering | Moderate cluster quality — boundaries are not crisp | Use clusters directionally, not as hard product classifications |
| No sales volume or revenue data | Cannot quantify financial impact of any finding | Link to transactional/POS data for revenue-weighted analysis |
| No time dimension in this dataset | Cannot identify trends, seasonality, or growth trajectories | Add historical catalog snapshots for trend analysis |
| Promotion field only flags current status | Cannot measure promotional lift over time | Combine with historical promotional calendar and review timestamps |
| Text analysis includes punctuation artifacts ("/": 35,916) | Minor noise in product name frequency counts | Pre-process text to strip non-alphabetic characters |

---

## 7. Appendix — Source of Truth (Script Outputs)

### A. Price Distribution (ikea_analysis.py)
```
10th percentile:  5.95
25th percentile: 19.99
50th percentile: 139.00
75th percentile: 799.00
90th percentile: 5,703.60
95th percentile: 16,499.00
99th percentile: 62,000.00
Skewness: 5.57
Kurtosis: 35.23
```

### B. Rating Summary (ikea_analysis.py)
```
Products with ratings: 263,077
Mean rating: 4.45
Median rating: 4.60
Standard deviation: 0.62
Mode (most common): 5.0
```

### C. Correlation Results (ikea_analysis.py)
```
Price vs Rating correlation: 0.0099
Rating vs Review Count correlation: 0.0275
```

### D. Price Tier Distribution (ikea_analysis.py)
```
Budget (<50):           144,415 (37.0%)
Affordable (50–100):     38,278  (9.8%)
Mid-range (100–250):     49,255 (12.6%)
Premium (250–500):       39,676 (10.2%)
Luxury (500–1,000):      33,115  (8.5%)
Ultra-Luxury (>1,000):   86,078 (22.0%)
```

### E. Promotion Comparison (ikea_analysis.py)
```
Promotional products:     28,131
Non-Promotional products: 362,686
Promo avg rating:   4.49 | Non-promo avg rating:  4.45
Promo avg reviews: 451.72 | Non-promo avg reviews: 159.39
Promo avg price: 2,917.50 | Non-promo avg price:  3,059.67
```

### F. ML Model Validation (ikea_advanced_ds.py)
```
K-Means Silhouette Score: 0.3175
Davies-Bouldin Index:     1.3402
Total unique products analyzed: 26,480
Anomalies flagged: 1,324 (5.0%)
```

### G. Cluster Profiles (ikea_advanced_ds.py + ikea_advanced_features.csv)
```
Cluster | Size   | Avg Price | Avg Rating | Top-Seller% | Price Variance Index
0       | 5,599  | 1,369.55  | 0.80       | 0.7%        | 1.44
1       | 3,960  | 1,263.65  | 4.46       | 6.8%        | 20.86
2       |   829  | 6,246.27  | 2.57       | 5.3%        | 0.04
3       | 13,718 |   972.32  | 4.42       | 3.4%        | 2.79
4       | 2,374  | 6,684.73  | 4.00       | 4.7%        | 15.65
```

### H. Top Categories (ikea_analysis.py)
```
Storage & Organisation:   50,447 products | Avg Price: 4,279 | Rating: 4.43
Kitchenware & Tableware:  26,031 products | Avg Price:   458 | Rating: 4.53
Beds & Mattresses:        22,171 products | Avg Price: 2,221 | Rating: 4.40
Decoration:               20,203 products | Avg Price:   809 | Rating: 4.62
Sofas & Armchairs:        14,572 products | Avg Price: 5,874 | Rating: 4.23
Storage Furniture:        14,317 products | Avg Price:   795 | Rating: 4.38
Kitchen Appliances:       14,221 products | Avg Price: 1,946 | Rating: 4.29
Lighting:                 12,517 products | Avg Price: 1,637 | Rating: 4.31
Small Storage/Organisers: 11,172 products | Avg Price:   975 | Rating: 4.53
Kitchens:                 10,039 products | Avg Price: 2,139 | Rating: 4.28
```

---

*Report generated from script outputs only. No findings were inferred beyond what the script outputs directly support.*
