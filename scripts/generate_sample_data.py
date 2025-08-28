#!/usr/bin/env python3
"""
Generate deterministic sample datasets for portfolio impact.
Creates realistic online retail data with proper distributions.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(n_rows=500, n_customers=150, days_span=90, output_file="online_retail_sample.csv"):
    """
    Generate realistic online retail sample data.
    
    Args:
        n_rows: Number of transactions
        n_customers: Number of unique customers
        days_span: Number of days to span
        output_file: Output filename
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Ensure we have enough customers
    n_customers = max(n_customers, n_rows // 4)
    
    # Generate customer IDs
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_span)
    
    # Create realistic customer behavior patterns
    # 20% high-value customers (more transactions, higher prices)
    # 60% regular customers (moderate transactions)
    # 20% occasional customers (few transactions)
    
    high_value_customers = customer_ids[:int(0.2 * n_customers)]
    regular_customers = customer_ids[int(0.2 * n_customers):int(0.8 * n_customers)]
    occasional_customers = customer_ids[int(0.8 * n_customers):]
    
    transactions = []
    
    # Generate transactions with realistic distributions
    for i in range(n_rows):
        # Assign customer based on behavior pattern
        if i < n_rows * 0.4:  # 40% of transactions from high-value customers
            customer_id = np.random.choice(high_value_customers)
        elif i < n_rows * 0.9:  # 50% from regular customers
            customer_id = np.random.choice(regular_customers)
        else:  # 10% from occasional customers
            customer_id = np.random.choice(occasional_customers)
        
        # Generate invoice date
        days_offset = np.random.randint(0, days_span)
        hours_offset = np.random.randint(0, 24)
        minutes_offset = np.random.randint(0, 60)
        
        invoice_date = start_date + timedelta(
            days=days_offset,
            hours=hours_offset,
            minutes=minutes_offset
        )
        
        # Generate invoice number
        invoice_no = f"INV-{invoice_date.strftime('%Y%m%d')}-{i:04d}"
        
        # Generate quantity and price based on customer type
        if customer_id in high_value_customers:
            # High-value customers: higher quantities and prices
            quantity = np.random.randint(1, 100)
            unit_price = round(np.random.uniform(10, 1000), 2)
        elif customer_id in regular_customers:
            # Regular customers: moderate quantities and prices
            quantity = np.random.randint(1, 50)
            unit_price = round(np.random.uniform(5, 200), 2)
        else:
            # Occasional customers: lower quantities and prices
            quantity = np.random.randint(1, 20)
            unit_price = round(np.random.uniform(1, 100), 2)
        
        transactions.append({
            'CustomerID': customer_id,
            'InvoiceDate': invoice_date.strftime('%Y-%m-%d %H:%M:%S'),
            'InvoiceNo': invoice_no,
            'Quantity': quantity,
            'UnitPrice': unit_price
        })
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(transactions)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def main():
    """Generate both sample and tiny datasets."""
    # Ensure data/sample directory exists
    os.makedirs('data/sample', exist_ok=True)
    
    # Generate main sample dataset
    print("Generating main sample dataset...")
    sample_df = generate_sample_data(n_rows=600, n_customers=150, days_span=90)
    sample_df.to_csv('data/sample/online_retail_sample.csv', index=False)
    
    # Generate tiny dataset for tests
    print("Generating tiny dataset...")
    tiny_df = generate_sample_data(n_rows=20, n_customers=8, days_span=30)
    tiny_df.to_csv('data/sample/online_retail_tiny.csv', index=False)
    
    # Print summary statistics
    print(f"\nSample dataset summary:")
    print(f"- Rows: {len(sample_df)}")
    print(f"- Unique customers: {sample_df['CustomerID'].nunique()}")
    print(f"- Date span: {sample_df['InvoiceDate'].min()} to {sample_df['InvoiceDate'].max()}")
    print(f"- Quantity range: {sample_df['Quantity'].min()} to {sample_df['Quantity'].max()}")
    print(f"- Price range: ${sample_df['UnitPrice'].min():.2f} to ${sample_df['UnitPrice'].max():.2f}")
    
    print(f"\nTiny dataset summary:")
    print(f"- Rows: {len(tiny_df)}")
    print(f"- Unique customers: {tiny_df['CustomerID'].nunique()}")
    
    print(f"\nFiles created:")
    print(f"- data/sample/online_retail_sample.csv")
    print(f"- data/sample/online_retail_tiny.csv")

if __name__ == "__main__":
    main()
