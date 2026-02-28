import pandas as pd
import numpy as np
from scipy import stats

# Load and clean data
print("Loading IKEA dataset...")
df = pd.read_csv('E:/Excel Datasets/IKEA_product_catalog.csv', header=None, low_memory=False)
df.columns = ['product_id_country', 'product_id', 'product_name', 'product_type', 
              'dimensions', 'description', 'category', 'subcategory', 
              'rating', 'review_count', 'promotion', 'available', 
              'product_url', 'price', 'currency', 'extra1', 'extra2', 'country']
df = df.iloc[1:].reset_index(drop=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce')
df_clean = df[(df['price'] > 0) & (df['price'] < 100000)].copy()

print('='*60)
print('TECHNIQUE 1: DISTRIBUTION ANALYSIS')
print('='*60)
print('Concept: Understanding how data is spread out')
print('Why: Shows us the typical values and variation in prices')
print()

# Price distribution
print('--- PRICE DISTRIBUTION ---')
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    value = df_clean['price'].quantile(p/100)
    print(f'{p}th percentile: {value:.2f}')

print()
print('Skewness:', round(df_clean['price'].skew(), 2))
print('Kurtosis:', round(df_clean['price'].kurtosis(), 2))
print()
print('EXPLANATION IN SIMPLE ENGLISH:')
print('- Skewness measures if prices are balanced or lopsided')
print('  Positive skew means most products are cheap, few are expensive')
print('- Kurtosis measures how extreme the outliers are')
print()

# Rating distribution
print('--- RATING DISTRIBUTION ---')
ratings_with_data = df_clean[df_clean['rating'].notna()]['rating']
print('Products with ratings:', len(ratings_with_data))
print('Mean rating:', round(ratings_with_data.mean(), 2))
print('Median rating:', round(ratings_with_data.median(), 2))
print('Standard deviation:', round(ratings_with_data.std(), 2))
print('Mode (most common):', ratings_with_data.mode().values[0])

rating_counts = ratings_with_data.value_counts().sort_index().head(10)
print('Rating breakdown:')
for rating, count in rating_counts.items():
    print(f'  {rating} stars: {count:,} products')

print()
print('='*60)
print('TECHNIQUE 2: CORRELATION ANALYSIS')
print('='*60)
print('Concept: Finding relationships between variables')
print('Why: Shows if higher prices lead to better ratings, etc.')
print()

# Correlation between price and rating
df_with_both = df_clean[(df_clean['price'].notna()) & (df_clean['rating'].notna())]
correlation = df_with_both['price'].corr(df_with_both['rating'])
print(f'Correlation between Price and Rating: {correlation:.4f}')
print()
print('EXPLANATION IN SIMPLE ENGLISH:')
print('- Correlation ranges from -1 to +1')
print('- Close to 0 = no relationship')
print('- Positive = as one increases, the other increases')
print('- Negative = as one increases, the other decreases')
print('- Our result shows almost NO relationship between price and rating')
print()

# Correlation between rating and review count
df_reviews = df_clean[(df_clean['rating'].notna()) & (df_clean['review_count'].notna())]
corr_reviews = df_reviews['rating'].corr(df_reviews['review_count'])
print(f'Correlation between Rating and Review Count: {corr_reviews:.4f}')
print('This shows if better-rated products get more reviews')

print()
print('='*60)
print('TECHNIQUE 3: CATEGORY CLUSTERING (Top Categories)')
print('='*60)
print('Concept: Grouping similar products together')
print('Why: Helps understand which product types perform similarly')
print()

# Analyze top categories
top_cats = df_clean['category'].value_counts().head(10)
print('Top10 Categories by Product Count:')
for cat, count in top_cats.items():
    avg_price = df_clean[df_clean['category'] == cat]['price'].mean()
    avg_rating = df_clean[(df_clean['category'] == cat) & (df_clean['rating'].notna())]['rating'].mean()
    print(f'  {cat}: {count:,} products | Avg Price: {avg_price:.2f} | Avg Rating: {avg_rating:.2f}')

print()
print('='*60)
print('TECHNIQUE 4: OUTLIER DETECTION')
print('='*60)
print('Concept: Finding unusual data points')
print('Why: Identifies products that are unusually expensive or cheap')
print()

# Price outliers using IQR method
Q1 = df_clean['price'].quantile(0.25)
Q3 = df_clean['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean['price'] < lower_bound) | (df_clean['price'] > upper_bound)]
print(f'Using IQR (Interquartile Range) method:')
print(f'  Q1 (25th percentile): {Q1:.2f}')
print(f'  Q3 (75th percentile): {Q3:.2f}')
print(f'  IQR: {IQR:.2f}')
print(f'  Lower bound: {lower_bound:.2f}')
print(f'  Upper bound: {upper_bound:.2f}')
print(f'  Number of outliers: {len(outliers):,} ({len(outliers)/len(df_clean)*100:.1f}%)')
print()
print('EXPLANATION IN SIMPLE ENGLISH:')
print('- IQR method: Outliers are values beyond 1.5x the middle spread')
print('- These are products with unusually high or low prices')
print()

# Top expensive outliers
print('Most Expensive Products (Outliers):')
expensive_outliers = outliers.nlargest(10, 'price')[['product_name', 'category', 'country', 'price', 'currency']]
print(expensive_outliers.to_string())

print()
print('='*60)
print('TECHNIQUE 5: GEOGRAPHIC ANALYSIS')
print('='*60)
print('Concept: Comparing products across different countries')
print('Why: Shows price variations and availability by region')
print()

# Price comparison across countries
country_stats = df_clean.groupby('country').agg({
    'price': ['mean', 'median', 'std'],
    'rating': 'mean',
    'product_id': 'count'
}).round(2)
country_stats.columns = ['avg_price', 'median_price', 'price_std', 'avg_rating', 'product_count']
country_stats = country_stats.sort_values('avg_price', ascending=False)
print('Countries by Average Price:')
print(country_stats.head(15).to_string())

print()
print('='*60)
print('TECHNIQUE 6: PROMOTION EFFECTIVENESS')
print('='*60)
print('Concept: Analyzing if promotions improve sales metrics')
print('Why: Shows if promoted products get better ratings/reviews')
print()

# Compare promoted vs non-promoted
promo_products = df_clean[df_clean['promotion'] != 'none']
non_promo = df_clean[df_clean['promotion'] == 'none']

print(f'Promotional Products: {len(promo_products):,}')
print(f'Non-Promotional Products: {len(non_promo):,}')
print()
print('Average Rating - Promotional:', round(promo_products['rating'].mean(), 2) if len(promo_products) > 0 else 'N/A')
print('Average Rating - Non-Promotional:', round(non_promo['rating'].mean(), 2) if len(non_promo) > 0 else 'N/A')
print()
print('Average Reviews - Promotional:', round(promo_products['review_count'].mean(), 2) if len(promo_products) > 0 else 'N/A')
print('Average Reviews - Non-Promotional:', round(non_promo['review_count'].mean(), 2) if len(non_promo) > 0 else 'N/A')
print()
print('Average Price - Promotional:', round(promo_products['price'].mean(), 2))
print('Average Price - Non-Promotional:', round(non_promo['price'].mean(), 2))
print()
print('EXPLANATION IN SIMPLE ENGLISH:')
print('- We compare products with promotions vs without')
print('- See if promotions correlate with better ratings or more reviews')

print()
print('='*60)
print('TECHNIQUE 7: TEXT ANALYSIS (Product Names)')
print('='*60)
print('Concept: Extracting common words from product names')
print('Why: Shows popular design themes and product focuses')
print()

# Word frequency in product names
all_names = ' '.join(df_clean['product_name'].dropna().astype(str))
words = all_names.lower().split()
word_freq = pd.Series(words).value_counts().head(20)
print('Most Common Words in Product Names:')
for word, count in word_freq.items():
    print(f'  {word}: {count:,}')

print()
print('EXPLANATION IN SIMPLE ENGLISH:')
print('- Shows what words IKEA uses most in naming products')
print('- Reveals design philosophy and product focus areas')

print()
print('='*60)
print('TECHNIQUE 8: PRICE TIER ANALYSIS')
print('='*60)
print('Concept: Segmenting products into price categories')
print('Why: Helps understand product positioning strategy')
print()

# Create price tiers
df_clean['price_tier'] = pd.cut(df_clean['price'], 
                                 bins=[0, 50, 100, 250, 500, 1000, float('inf')],
                                 labels=['Budget (<50)', 'Affordable (50-100)', 
                                        'Mid-range (100-250)', 'Premium (250-500)', 
                                        'Luxury (500-1000)', 'Ultra-Luxury (>1000)'])

tier_counts = df_clean['price_tier'].value_counts().sort_index()
print('Product Distribution by Price Tier:')
for tier, count in tier_counts.items():
    pct = count / len(df_clean) * 100
    print(f'  {tier}: {count:,} products ({pct:.1f}%)')

print()
avg_rating_by_tier = df_clean.groupby('price_tier')['rating'].mean()
print('Average Rating by Price Tier:')
for tier, rating in avg_rating_by_tier.items():
    if pd.notna(rating):
        print(f'  {tier}: {rating:.2f}')

print()
print('EXPLANATION IN SIMPLE ENGLISH:')
print('- Segments products into price categories')
print('- Shows if more expensive products have better ratings')
print('- Reveals IKEA product portfolio distribution')

print()
print('='*60)
print('ANALYSIS COMPLETE!')
print('='*60)
