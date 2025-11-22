#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
# ]
# ///

"""
Simple data preprocessing and feature engineering pipeline.
Demonstrates: handling missing values, scaling, encoding, and feature creation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Create synthetic dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'category': np.random.choice(['A', 'B', 'C'], n_samples),
    'has_car': np.random.choice([0, 1], n_samples),
    'target': np.random.choice([0, 1], n_samples)
})

# Introduce missing values
data.loc[np.random.choice(data.index, 50), 'income'] = np.nan
data.loc[np.random.choice(data.index, 30), 'credit_score'] = np.nan

print("Data Preprocessing Pipeline")
print("=" * 60)
print(f"\nOriginal dataset shape: {data.shape}")
print(f"\nMissing values:\n{data.isnull().sum()}")
print(f"\nFirst few rows:\n{data.head()}")

# 1. Handle missing values
print("\n[1] Handling Missing Values")
imputer_median = SimpleImputer(strategy='median')
data['income'] = imputer_median.fit_transform(data[['income']])
data['credit_score'] = imputer_median.fit_transform(data[['credit_score']])
print(f"    Missing values after imputation: {data.isnull().sum().sum()}")

# 2. Feature Engineering
print("\n[2] Feature Engineering")
# Create new features
data['income_to_age_ratio'] = data['income'] / data['age']
data['income_category'] = pd.cut(data['income'], bins=3, labels=['low', 'medium', 'high'])
data['age_group'] = pd.cut(data['age'], bins=[0, 30, 50, 100], labels=['young', 'middle', 'senior'])
print(f"    Created 3 new features")
print(f"    New shape: {data.shape}")

# 3. Encode categorical variables
print("\n[3] Encoding Categorical Variables")
# Label encoding for ordinal features
label_encoder = LabelEncoder()
data['income_category_encoded'] = label_encoder.fit_transform(data['income_category'].astype(str))
data['age_group_encoded'] = label_encoder.fit_transform(data['age_group'].astype(str))

# One-hot encoding for nominal features
data_encoded = pd.get_dummies(data, columns=['category'], prefix='cat')
print(f"    Shape after encoding: {data_encoded.shape}")

# 4. Feature Scaling
print("\n[4] Feature Scaling")
# Select numerical features to scale
numerical_features = ['age', 'income', 'credit_score', 'income_to_age_ratio']
scaler = StandardScaler()
data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])
print(f"    Scaled {len(numerical_features)} numerical features")

# 5. Split data
print("\n[5] Train-Test Split")
feature_cols = [col for col in data_encoded.columns if col not in ['target', 'income_category', 'age_group']]
X = data_encoded[feature_cols]
y = data_encoded['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"    Train set: {X_train.shape}")
print(f"    Test set: {X_test.shape}")

# Summary
print("\n" + "=" * 60)
print("Pipeline Summary:")
print(f"  - Original features: 6")
print(f"  - Final features: {X_train.shape[1]}")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")
print(f"\nFeatures: {list(X.columns)}")
