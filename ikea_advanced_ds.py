import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
# import matplotlib.pyplot as plt (unused)
# import seaborn as sns (unused)

# Load data
print("Loading IKEA dataset...")
df = pd.read_csv('e:/Excel Datasets/IKEA_product_catalog.csv')

# Clean numeric columns before aggregation
print("Cleaning data and converting to numeric...")
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['product_rating'] = pd.to_numeric(df['product_rating'], errors='coerce')
df['product_rating_count'] = pd.to_numeric(df['product_rating_count'], errors='coerce')

# --- 1. PRODUCT-LEVEL FEATURE ENGINEERING ---
print("Grouping by product_id for global metrics...")

# Global price stats per product
global_stats = df.groupby('product_id').agg({
    'price': ['mean', 'min', 'max', 'std', 'count'],
    'product_rating': 'mean',
    'product_rating_count': 'sum'
}).reset_index()

global_stats.columns = ['product_id', 'avg_price', 'min_price', 'max_price', 'price_std', 'country_count', 'avg_rating', 'total_reviews']

# Price Variance Index (How much does price fluctuate globally?)
global_stats['price_variance_index'] = (global_stats['max_price'] - global_stats['min_price']) / (global_stats['avg_price'] + 1e-6)

# Badge Density (How often is it a top seller?)
df['is_top_seller'] = df['badge'].apply(lambda x: 1 if x == 'TOP_SELLER' else 0)
top_seller_density = df.groupby('product_id')['is_top_seller'].mean().reset_index()
top_seller_density.rename(columns={'is_top_seller': 'top_seller_pct'}, inplace=True)

# Merge back
product_features = pd.merge(global_stats, top_seller_density, on='product_id')

# --- 2. MEASUREMENT EXTRACTION (Regex) ---
def extract_dimensions(dim_str):
    if pd.isna(dim_str) or dim_str == 'none':
        return pd.Series([np.nan, np.nan, np.nan])
    # Match patterns like 50x27x36 cm or 50 cm
    nums = re.findall(r'(\d+(?:\.\d+)?)', str(dim_str))
    nums = [float(n) for n in nums]
    if len(nums) >= 3:
        return pd.Series([nums[0], nums[1], nums[2]])
    elif len(nums) == 2:
        return pd.Series([nums[0], nums[1], np.nan])
    elif len(nums) == 1:
        return pd.Series([nums[0], np.nan, np.nan])
    return pd.Series([np.nan, np.nan, np.nan])

print("Extracting physical dimensions...")
# Get dimensions from the first occurrence of each product
dim_df = df.drop_duplicates('product_id')[['product_id', 'product_measurements']]
dim_extracted = dim_df['product_measurements'].apply(extract_dimensions)
dim_extracted.columns = ['dim1', 'dim2', 'dim3']
dim_extracted['product_id'] = dim_df['product_id'].values

product_features = pd.merge(product_features, dim_extracted, on='product_id')

# Calculate Volume and Aspect Ratio
product_features['volume_est'] = product_features['dim1'] * product_features['dim2'] * product_features['dim3']
product_features['aspect_ratio'] = product_features['dim1'] / (product_features['dim2'] + 1e-6)

# Clean features for ML
product_features = product_features.fillna(0)
# Filter obvious outliers in price for stability (reduces noise)
product_features = product_features[product_features['avg_price'] < 10000].copy()

ml_features = product_features.drop(columns=['product_id'])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(ml_features)

# --- 3. CLUSTERING ---
print("Running K-Means...")
kmeans = KMeans(n_clusters=5, random_state=42)
product_features['cluster_kmeans'] = kmeans.fit_predict(X_scaled)

print("Running GMM...")
gmm = GaussianMixture(n_components=5, random_state=42)
product_features['cluster_gmm'] = gmm.fit_predict(X_scaled)

# --- 4. MACHINE LEARNING: ANOMALY DETECTION ---
print("Running Anomaly Detection (Isolation Forest)...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
product_features['is_anomaly'] = iso_forest.fit_predict(X_scaled)
# -1 is anomaly, 1 is normal -> convert to 1 for anomaly
product_features['is_anomaly'] = product_features['is_anomaly'].apply(lambda x: 1 if x == -1 else 0)

# --- 5. VALIDATION ---
sil = silhouette_score(X_scaled, product_features['cluster_kmeans'])
db = davies_bouldin_score(X_scaled, product_features['cluster_kmeans'])
print(f"K-Means Silhouette Score: {sil:.4f}")
print(f"Davies-Bouldin Index: {db:.4f}")

# --- 6. RESULTS PREVIEW ---
print("\nAnomaly Sample (Top 5):")
print(product_features[product_features['is_anomaly'] == 1].head())

print("\nCluster Profiling (Average Price per Cluster):")
print(product_features.groupby('cluster_kmeans')['avg_price'].mean())

# Save enhanced features
product_features.to_csv('e:/Excel Datasets/ikea_advanced_features.csv', index=False)
print("\nSuccess! Advanced features and clusters saved to 'ikea_advanced_features.csv'")
