# IKEA Global Intelligence: Advanced ML & 3D Analytics 🌌📊

## 1. Executive Summary
This project transforms a raw catalog of 400,000+ IKEA product listings into a strategic business intelligence engine. By leveraging **Unsupervised Machine Learning**, **Anomaly Detection**, and **High-Dimensional Visualizations**, we identify hidden product archetypes and the key drivers of global popularity.

---

## 2. Technical Methodology
### A. Advanced Feature Engineering
*   **Global Normalization**: Aggregated product pricing across 50+ countries to identify "Global Staples" vs "Regional Elasticities."
*   **Physical DNA Extraction**: Used **Regex** to parse dimensions and calculate **Estimated Volume** and **Aspect Ratio**.
*   **Success Metrics**: Engineered a `Top_Seller_Pct` feature to measure a product's global hit rate.

### B. Machine Learning Architecture
*   **Clustering (K-Means & GMM)**: Segmented the portfolio into 5 distinct business archetypes (Budget, Basics, Premium, etc.).
*   **Anomaly Detection (Isolation Forest)**: Automatically flagged unique "Oddballs"—high-potential niche products or pricing errors.
*   **Predictive Modeling (XGBoost)**: Trained a regressor to identify popularity drivers. Key Insight: **Reviews and Price Consistency move the needle more than deep discounts.**

---

## 3. Business Strategy: The "Human" Perspective
We've categorized the IKEA universe into five actionable groups:

1.  **Budget Entry (High Volume)**: Low price, high sales. *Keep these front-and-center to drive foot traffic.*
2.  **Reliable Basics**: Mid-range, stable prices. *The "bread and butter" of the revenue stream.*
3.  **Premium Flagships**: High price, high weight/volume. *Requires high-touch marketing and premium store placement.*
4.  **Mid-Range Statement**: Stylish, aesthetic pieces. *Primary target for social media/influencer campaigns.*
5.  **The Oddballs (Anomalies)**: Unique items. *Investigate these for potential new market segments.*

---

## 4. How to Read the Interactive Visuals
*   **The 3D Product Galaxy**: Each dot is a product. 
    *   **Higher** = Larger item.
    *   **Further Right** = More expensive.
    *   **Color** = The Business Archetype.
*   **The Strategic Radar Chart**: Shows the "DNA" of each cluster. A wider shape means a more "balanced" and successful product category.

---

## 5. Conclusion & Recommendations
*   **Strategy**: Focus on products with 80+ reviews to push them over the "Success Threshold" of 100.
*   **Pricing**: Standardize global pricing for flagship items to improve trust and popularity.
*   **Innovation**: Use Anomaly detection to find the next "Viral" product before it goes mainstream.

---
*Created with 💙 by Antigravity x Terry Mathew*
