#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "pandas",
#     "numpy",
# ]
# ///

"""
Simple data visualization dashboard using matplotlib and seaborn.
Demonstrates: various plot types, statistical visualizations, and data exploration.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("Data Visualization Dashboard")
print("=" * 70)

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.lognormal(10.5, 0.5, n_samples),
    'score': np.random.normal(70, 15, n_samples),
    'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
    'satisfied': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
})

# Clip scores to valid range
data['score'] = data['score'].clip(0, 100)

print(f"Dataset: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"\nFirst few rows:")
print(data.head())

print("\nDataset Statistics:")
print(data.describe())

# Create visualization dashboard
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Data Visualization Dashboard', fontsize=16, fontweight='bold')

# 1. Histogram
ax1 = plt.subplot(3, 3, 1)
plt.hist(data['age'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# 2. Box plot
ax2 = plt.subplot(3, 3, 2)
data.boxplot(column='income', by='category', ax=ax2)
plt.title('Income by Category')
plt.suptitle('')  # Remove automatic title
plt.xlabel('Category')
plt.ylabel('Income ($)')

# 3. Scatter plot
ax3 = plt.subplot(3, 3, 3)
scatter = plt.scatter(data['age'], data['income'], c=data['score'],
                     cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Score')
plt.title('Age vs Income (colored by Score)')
plt.xlabel('Age')
plt.ylabel('Income ($)')

# 4. Bar chart
ax4 = plt.subplot(3, 3, 4)
category_counts = data['category'].value_counts()
plt.bar(category_counts.index, category_counts.values, edgecolor='black')
plt.title('Count by Category')
plt.xlabel('Category')
plt.ylabel('Count')

# 5. Violin plot
ax5 = plt.subplot(3, 3, 5)
sns.violinplot(data=data, x='region', y='score', ax=ax5)
plt.title('Score Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Score')
plt.xticks(rotation=45)

# 6. Heatmap (correlation matrix)
ax6 = plt.subplot(3, 3, 6)
numeric_cols = ['age', 'income', 'score', 'satisfied']
corr_matrix = data[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
           center=0, ax=ax6, cbar_kws={'label': 'Correlation'})
plt.title('Correlation Matrix')

# 7. Line plot (grouped)
ax7 = plt.subplot(3, 3, 7)
grouped = data.groupby('category')['score'].agg(['mean', 'std']).reset_index()
x_pos = np.arange(len(grouped))
plt.bar(x_pos, grouped['mean'], yerr=grouped['std'],
       capsize=5, alpha=0.7, edgecolor='black')
plt.xticks(x_pos, grouped['category'])
plt.title('Average Score by Category (with std)')
plt.xlabel('Category')
plt.ylabel('Average Score')

# 8. Pie chart
ax8 = plt.subplot(3, 3, 8)
satisfied_counts = data['satisfied'].value_counts()
plt.pie(satisfied_counts.values, labels=['Not Satisfied', 'Satisfied'],
       autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#99ff99'])
plt.title('Customer Satisfaction')

# 9. Stacked bar chart
ax9 = plt.subplot(3, 3, 9)
pivot_data = pd.crosstab(data['region'], data['satisfied'])
pivot_data.plot(kind='bar', stacked=True, ax=ax9,
               color=['#ff9999', '#99ff99'])
plt.title('Satisfaction by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.legend(['Not Satisfied', 'Satisfied'])
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('data_visualization_dashboard.png', dpi=150, bbox_inches='tight')
print("\n✓ Dashboard saved as 'data_visualization_dashboard.png'")

# Additional statistical visualizations
print("\n" + "=" * 70)
print("Creating Additional Visualizations...")

# Create a second figure with more advanced plots
fig2 = plt.figure(figsize=(16, 6))
fig2.suptitle('Advanced Statistical Visualizations', fontsize=16, fontweight='bold')

# 1. KDE plot
ax1 = plt.subplot(1, 3, 1)
for category in data['category'].unique():
    subset = data[data['category'] == category]['score']
    sns.kdeplot(subset, label=category, ax=ax1)
plt.title('Score Distribution by Category (KDE)')
plt.xlabel('Score')
plt.ylabel('Density')
plt.legend()

# 2. Joint plot data (scatter + histograms)
ax2 = plt.subplot(1, 3, 2)
plt.scatter(data['age'], data['score'], alpha=0.5)
plt.title('Age vs Score')
plt.xlabel('Age')
plt.ylabel('Score')

# Add trend line
z = np.polyfit(data['age'], data['score'], 1)
p = np.poly1d(z)
plt.plot(data['age'].sort_values(), p(data['age'].sort_values()),
        "r--", alpha=0.8, label=f'Trend line')
plt.legend()

# 3. Count plot with hue
ax3 = plt.subplot(1, 3, 3)
satisfaction_by_category = pd.crosstab(data['category'], data['satisfied'])
satisfaction_by_category.plot(kind='bar', ax=ax3, color=['#ff9999', '#99ff99'])
plt.title('Satisfaction Count by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.legend(['Not Satisfied', 'Satisfied'])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('advanced_visualizations.png', dpi=150, bbox_inches='tight')
print("✓ Advanced visualizations saved as 'advanced_visualizations.png'")

print("\n" + "=" * 70)
print("Visualization dashboard completed!")
print("\nGenerated files:")
print("  - data_visualization_dashboard.png")
print("  - advanced_visualizations.png")
