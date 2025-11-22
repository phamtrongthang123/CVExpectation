#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
# ]
# ///

"""
Simple time series analysis and forecasting.
Demonstrates: time series decomposition, trend analysis, and simple forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

print("Time Series Analysis Tool")
print("=" * 70)

# ============================================================================
# 1. Generate synthetic time series data
# ============================================================================
print("\n[1] Generating Time Series Data")

# Create daily data for 2 years
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
n_days = len(dates)

# Components of the time series
np.random.seed(42)

# Trend (upward)
trend = np.linspace(100, 200, n_days)

# Seasonality (yearly and weekly)
yearly_seasonality = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365)
weekly_seasonality = 5 * np.sin(2 * np.pi * np.arange(n_days) / 7)

# Random noise
noise = np.random.normal(0, 5, n_days)

# Combine components
values = trend + yearly_seasonality + weekly_seasonality + noise

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'value': values
})
df.set_index('date', inplace=True)

print(f"   Generated {len(df)} daily observations")
print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
print(f"\n   First few rows:")
print(df.head())

# ============================================================================
# 2. Time Series Statistics
# ============================================================================
print("\n[2] Time Series Statistics")
print("   " + "-" * 66)
print(f"   Mean:                {df['value'].mean():.2f}")
print(f"   Std Dev:             {df['value'].std():.2f}")
print(f"   Min:                 {df['value'].min():.2f}")
print(f"   Max:                 {df['value'].max():.2f}")
print(f"   25th percentile:     {df['value'].quantile(0.25):.2f}")
print(f"   Median:              {df['value'].median():.2f}")
print(f"   75th percentile:     {df['value'].quantile(0.75):.2f}")
print("   " + "-" * 66)

# ============================================================================
# 3. Moving Averages
# ============================================================================
print("\n[3] Calculating Moving Averages")

df['MA_7'] = df['value'].rolling(window=7).mean()
df['MA_30'] = df['value'].rolling(window=30).mean()
df['MA_90'] = df['value'].rolling(window=90).mean()

print("   ✓ Calculated 7-day moving average")
print("   ✓ Calculated 30-day moving average")
print("   ✓ Calculated 90-day moving average")

# ============================================================================
# 4. Trend Analysis
# ============================================================================
print("\n[4] Trend Analysis")

# Calculate daily change
df['daily_change'] = df['value'].diff()
df['daily_change_pct'] = df['value'].pct_change() * 100

# Calculate trend using linear regression
from numpy.polynomial import Polynomial
x = np.arange(len(df))
y = df['value'].values
p = Polynomial.fit(x, y, 1)
df['trend_line'] = p(x)

print(f"   Average daily change: {df['daily_change'].mean():.2f}")
print(f"   Average daily change (%): {df['daily_change_pct'].mean():.2f}%")
print(f"   Trend coefficient: {p.coef[1]:.4f}")

# ============================================================================
# 5. Seasonality Detection
# ============================================================================
print("\n[5] Seasonality Analysis")

# Monthly aggregation
df_monthly = df.resample('M')['value'].mean()
print(f"   Monthly average:")
for month_date, avg_value in df_monthly.head(6).items():
    print(f"      {month_date.strftime('%Y-%m'):10s}: {avg_value:.2f}")

# Day of week pattern
df['day_of_week'] = df.index.dayofweek
weekly_pattern = df.groupby('day_of_week')['value'].mean()
print(f"\n   Weekly pattern (Mon=0, Sun=6):")
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for day_num, avg_value in weekly_pattern.items():
    print(f"      {days[day_num]:3s}: {avg_value:.2f}")

# ============================================================================
# 6. Simple Forecasting (Moving Average)
# ============================================================================
print("\n[6] Simple Forecasting")

# Use last 30 days average for forecast
forecast_days = 30
last_30_days_avg = df['value'].tail(30).mean()

forecast_dates = pd.date_range(
    start=df.index.max() + timedelta(days=1),
    periods=forecast_days,
    freq='D'
)

forecast_df = pd.DataFrame({
    'date': forecast_dates,
    'forecast': last_30_days_avg
})
forecast_df.set_index('date', inplace=True)

print(f"   Forecast for next {forecast_days} days:")
print(f"   Predicted value (30-day MA): {last_30_days_avg:.2f}")

# ============================================================================
# 7. Anomaly Detection
# ============================================================================
print("\n[7] Anomaly Detection")

# Simple threshold-based anomaly detection
mean = df['value'].mean()
std = df['value'].std()
threshold = 2.5  # 2.5 standard deviations

df['is_anomaly'] = np.abs((df['value'] - mean) / std) > threshold
anomalies = df[df['is_anomaly']]

print(f"   Detected {len(anomalies)} anomalies (>{threshold} std from mean)")
if len(anomalies) > 0:
    print(f"   First few anomalies:")
    for date, row in anomalies.head(3).iterrows():
        print(f"      {date.date()}: {row['value']:.2f} (z-score: {(row['value']-mean)/std:.2f})")

# ============================================================================
# 8. Visualization
# ============================================================================
print("\n[8] Creating Visualizations")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Time Series Analysis Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Original data with moving averages
axes[0].plot(df.index, df['value'], label='Original', alpha=0.6, linewidth=0.8)
axes[0].plot(df.index, df['MA_7'], label='7-day MA', linewidth=1.5)
axes[0].plot(df.index, df['MA_30'], label='30-day MA', linewidth=1.5)
axes[0].plot(df.index, df['trend_line'], label='Trend', linewidth=2, linestyle='--')
axes[0].scatter(anomalies.index, anomalies['value'], color='red', s=50,
               label='Anomalies', zorder=5)
axes[0].set_title('Time Series with Moving Averages and Trend')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Daily changes
axes[1].plot(df.index, df['daily_change'], linewidth=0.8)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
axes[1].set_title('Daily Changes')
axes[1].set_ylabel('Change')
axes[1].grid(True, alpha=0.3)

# Plot 3: Distribution
axes[2].hist(df['value'], bins=50, edgecolor='black', alpha=0.7)
axes[2].axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
axes[2].axvline(mean + 2*std, color='orange', linestyle='--', linewidth=1, label=f'±2σ')
axes[2].axvline(mean - 2*std, color='orange', linestyle='--', linewidth=1)
axes[2].set_title('Value Distribution')
axes[2].set_xlabel('Value')
axes[2].set_ylabel('Frequency')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_series_analysis.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved visualization: time_series_analysis.png")

print("\n" + "=" * 70)
print("Time Series Analysis completed!")
