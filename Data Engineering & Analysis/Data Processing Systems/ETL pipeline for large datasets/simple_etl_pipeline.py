#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pandas",
#     "numpy",
# ]
# ///

"""
Simple ETL (Extract, Transform, Load) pipeline for data processing.
Demonstrates: data extraction, transformation, validation, and loading.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

print("ETL Pipeline Demo")
print("=" * 70)

# ============================================================================
# EXTRACT: Load data from multiple sources
# ============================================================================
print("\n[1] EXTRACT: Loading data from multiple sources")

# Source 1: Generate CSV data
print("   Loading from CSV source...")
csv_data = pd.DataFrame({
    'user_id': range(1, 1001),
    'signup_date': [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(1000)],
    'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia'], 1000),
})
csv_data.to_csv('temp_users.csv', index=False)
df_users = pd.read_csv('temp_users.csv')
print(f"   ✓ Loaded {len(df_users)} user records from CSV")

# Source 2: Generate JSON data
print("   Loading from JSON source...")
transactions_data = [
    {
        'transaction_id': i,
        'user_id': np.random.randint(1, 1001),
        'amount': round(np.random.uniform(10, 1000), 2),
        'timestamp': (datetime.now() - timedelta(hours=np.random.randint(1, 720))).isoformat()
    }
    for i in range(1, 5001)
]
with open('temp_transactions.json', 'w') as f:
    json.dump(transactions_data, f)

with open('temp_transactions.json', 'r') as f:
    transactions = json.load(f)
df_transactions = pd.DataFrame(transactions)
print(f"   ✓ Loaded {len(df_transactions)} transactions from JSON")

# Source 3: Simulated database data
print("   Loading from database...")
df_products = pd.DataFrame({
    'product_id': range(1, 101),
    'product_name': [f'Product_{i}' for i in range(1, 101)],
    'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
    'price': np.random.uniform(10, 500, 100).round(2)
})
print(f"   ✓ Loaded {len(df_products)} products from database")

# ============================================================================
# TRANSFORM: Clean, validate, and transform data
# ============================================================================
print("\n[2] TRANSFORM: Cleaning and transforming data")

# 2.1: Data Cleaning
print("   [2.1] Data Cleaning")

# Convert data types
df_users['signup_date'] = pd.to_datetime(df_users['signup_date'])
df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'])
print("   ✓ Converted date columns to datetime")

# Handle missing values (simulate some missing data first)
df_transactions.loc[df_transactions.sample(50).index, 'amount'] = np.nan
missing_before = df_transactions['amount'].isna().sum()
df_transactions['amount'].fillna(df_transactions['amount'].median(), inplace=True)
print(f"   ✓ Filled {missing_before} missing values in transaction amounts")

# Remove duplicates
duplicates = df_transactions.duplicated(subset=['user_id', 'timestamp']).sum()
df_transactions.drop_duplicates(subset=['user_id', 'timestamp'], inplace=True)
print(f"   ✓ Removed {duplicates} duplicate transactions")

# 2.2: Data Validation
print("   [2.2] Data Validation")

# Validate transaction amounts
invalid_amounts = (df_transactions['amount'] <= 0).sum()
df_transactions = df_transactions[df_transactions['amount'] > 0]
print(f"   ✓ Removed {invalid_amounts} invalid transactions (amount <= 0)")

# Validate user IDs exist
invalid_users = ~df_transactions['user_id'].isin(df_users['user_id'])
removed_count = invalid_users.sum()
df_transactions = df_transactions[~invalid_users]
print(f"   ✓ Removed {removed_count} transactions with invalid user_id")

# 2.3: Feature Engineering
print("   [2.3] Feature Engineering")

# Add derived columns
df_transactions['date'] = df_transactions['timestamp'].dt.date
df_transactions['hour'] = df_transactions['timestamp'].dt.hour
df_transactions['day_of_week'] = df_transactions['timestamp'].dt.dayofweek
print("   ✓ Created temporal features: date, hour, day_of_week")

# User aggregations
user_stats = df_transactions.groupby('user_id').agg({
    'amount': ['sum', 'mean', 'count'],
    'transaction_id': 'count'
}).reset_index()
user_stats.columns = ['user_id', 'total_spent', 'avg_transaction', 'transaction_count', 'num_transactions']
print("   ✓ Created user aggregation features")

# 2.4: Data Enrichment (Join with user data)
print("   [2.4] Data Enrichment")
df_enriched = df_transactions.merge(df_users, on='user_id', how='left')
print(f"   ✓ Enriched transactions with user data")

# Calculate user tenure
df_enriched['user_tenure_days'] = (pd.Timestamp.now() - df_enriched['signup_date']).dt.days
print("   ✓ Calculated user tenure")

# ============================================================================
# LOAD: Save transformed data
# ============================================================================
print("\n[3] LOAD: Saving transformed data")

# Create output directory
output_dir = 'etl_output'
os.makedirs(output_dir, exist_ok=True)

# Save to different formats
df_enriched.to_csv(f'{output_dir}/enriched_transactions.csv', index=False)
print(f"   ✓ Saved enriched data to CSV: {len(df_enriched)} rows")

user_stats.to_json(f'{output_dir}/user_statistics.json', orient='records', indent=2)
print(f"   ✓ Saved user statistics to JSON: {len(user_stats)} users")

# Save summary statistics
summary = {
    'pipeline_run_time': datetime.now().isoformat(),
    'total_transactions': len(df_enriched),
    'total_users': df_enriched['user_id'].nunique(),
    'total_revenue': float(df_enriched['amount'].sum()),
    'avg_transaction_value': float(df_enriched['amount'].mean()),
    'date_range': {
        'start': df_enriched['timestamp'].min().isoformat(),
        'end': df_enriched['timestamp'].max().isoformat()
    }
}

with open(f'{output_dir}/pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"   ✓ Saved pipeline summary")

# ============================================================================
# MONITORING: Pipeline statistics
# ============================================================================
print("\n[4] PIPELINE MONITORING")
print("   " + "-" * 66)
print(f"   Records processed:      {len(df_enriched):,}")
print(f"   Unique users:           {df_enriched['user_id'].nunique():,}")
print(f"   Total revenue:          ${df_enriched['amount'].sum():,.2f}")
print(f"   Avg transaction:        ${df_enriched['amount'].mean():.2f}")
print(f"   Date range:             {df_enriched['timestamp'].min().date()} to {df_enriched['timestamp'].max().date()}")
print("   " + "-" * 66)

# Top countries by revenue
print("\n   Top 3 Countries by Revenue:")
country_revenue = df_enriched.groupby('country')['amount'].sum().sort_values(ascending=False)
for i, (country, revenue) in enumerate(country_revenue.head(3).items(), 1):
    print(f"      {i}. {country:15s} ${revenue:,.2f}")

# Cleanup temp files
os.remove('temp_users.csv')
os.remove('temp_transactions.json')

print("\n" + "=" * 70)
print("ETL Pipeline completed successfully!")
print(f"Output files saved to: {output_dir}/")
