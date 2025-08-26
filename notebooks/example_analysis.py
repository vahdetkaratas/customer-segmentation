"""
Example Customer Segmentation Analysis
This script demonstrates the RFM analysis process step by step
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import our RFM analysis functions
import sys
sys.path.append('../src')
from rfm_analysis import create_sample_dataset, clean_dataset, calculate_rfm_metrics

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("CUSTOMER SEGMENTATION - EXAMPLE ANALYSIS")
print("=" * 60)

# Step 1: Load or generate data
print("\n1. Loading data...")
try:
    # Try to load existing data
    df = pd.read_csv('../data/raw/sample_transactions.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print("✓ Loaded existing dataset")
except FileNotFoundError:
    # Generate new data if not available
    print("Generating new sample dataset...")
    df = create_sample_dataset()
    df.to_csv('../data/raw/sample_transactions.csv', index=False)

print(f"   Dataset shape: {df.shape}")
print(f"   Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print(f"   Unique customers: {df['CustomerID'].nunique()}")

# Step 2: Data cleaning
print("\n2. Cleaning data...")
df_clean = clean_dataset(df)

# Step 3: Calculate RFM metrics
print("\n3. Calculating RFM metrics...")
rfm_table = calculate_rfm_metrics(df_clean)

# Step 4: Display results
print("\n4. RFM Analysis Results:")
print("-" * 40)
print("First 10 customers:")
print(rfm_table.head(10).to_string(index=False))

print("\nSummary Statistics:")
print(rfm_table[['Recency', 'Frequency', 'Monetary']].describe())

# Step 5: Basic visualizations
print("\n5. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Recency distribution
axes[0,0].hist(rfm_table['Recency'], bins=30, alpha=0.7, color='lightblue')
axes[0,0].set_title('Recency Distribution')
axes[0,0].set_xlabel('Days Since Last Purchase')

# Frequency distribution
axes[0,1].hist(rfm_table['Frequency'], bins=20, alpha=0.7, color='lightgreen')
axes[0,1].set_title('Frequency Distribution')
axes[0,1].set_xlabel('Number of Purchases')

# Monetary distribution
axes[1,0].hist(rfm_table['Monetary'], bins=30, alpha=0.7, color='lightcoral')
axes[1,0].set_title('Monetary Distribution')
axes[1,0].set_xlabel('Total Spending ($)')

# Scatter plot: Recency vs Frequency
axes[1,1].scatter(rfm_table['Recency'], rfm_table['Frequency'], alpha=0.6)
axes[1,1].set_title('Recency vs Frequency')
axes[1,1].set_xlabel('Recency (Days)')
axes[1,1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('../reports/figures/rfm_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Analysis completed successfully!")
print("✓ Visualizations saved to ../reports/figures/")
print("✓ RFM table saved to ../data/processed/rfm_table.csv")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("1. Run clustering analysis on RFM scores")
print("2. Create customer segments")
print("3. Develop segment-specific strategies")
print("4. Build interactive dashboard")
print("=" * 60)
