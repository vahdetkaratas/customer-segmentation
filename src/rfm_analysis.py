"""
Customer Segmentation Project - RFM Analysis
First Step: Data Generation and RFM Metrics Calculation
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)


def create_sample_dataset(n_customers=1000, n_transactions=5000):
    """
    Generate a reproducible sample customer transaction dataset

    Parameters:
    - n_customers: Number of unique customers
    - n_transactions: Total number of transactions

    Returns:
    - DataFrame with columns: CustomerID, InvoiceDate, InvoiceNo, Quantity, UnitPrice
    """
    print("Generating sample customer transaction dataset...")

    # Generate customer IDs
    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]

    # Generate transaction data
    data = []

    # Date range: 2 years of data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 1, 1)

    for i in range(n_transactions):
        # Random customer
        customer_id = np.random.choice(customer_ids)

        # Random date within range
        random_days = np.random.randint(0, (end_date - start_date).days)
        invoice_date = start_date + timedelta(days=random_days)

        # Random invoice number
        invoice_no = f"INV_{i:06d}"

        # Random quantity (1-50, with some outliers)
        if np.random.random() < 0.95:  # 95% normal quantities
            quantity = np.random.randint(1, 51)
        else:  # 5% larger quantities
            quantity = np.random.randint(51, 200)

        # Random unit price (5-500, with some outliers)
        if np.random.random() < 0.90:  # 90% normal prices
            unit_price = round(np.random.uniform(5, 500), 2)
        else:  # 10% higher prices
            unit_price = round(np.random.uniform(500, 2000), 2)

        data.append(
            {
                "CustomerID": customer_id,
                "InvoiceDate": invoice_date,
                "InvoiceNo": invoice_no,
                "Quantity": quantity,
                "UnitPrice": unit_price,
            }
        )

    df = pd.DataFrame(data)
    print(
        f"Generated {len(df)} transactions for {df['CustomerID'].nunique()} unique customers"
    )
    return df


def clean_dataset(df):
    """
    Clean the dataset by removing missing values and outliers

    Parameters:
    - df: Raw transaction DataFrame

    Returns:
    - Cleaned DataFrame
    """
    print("Cleaning dataset...")

    # Remove rows with missing values
    initial_rows = len(df)
    df_clean = df.dropna()
    print(f"Removed {initial_rows - len(df_clean)} rows with missing values")

    # Remove negative or zero quantities and prices
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]
    print(f"Removed {len(df) - len(df_clean)} rows with invalid quantities/prices")

    # Remove extreme outliers (optional - you can adjust these thresholds)
    q1_qty, q3_qty = df_clean["Quantity"].quantile([0.25, 0.75])
    iqr_qty = q3_qty - q1_qty
    qty_upper = q3_qty + 1.5 * iqr_qty

    q1_price, q3_price = df_clean["UnitPrice"].quantile([0.25, 0.75])
    iqr_price = q3_price - q1_price
    price_upper = q3_price + 1.5 * iqr_price

    df_clean = df_clean[
        (df_clean["Quantity"] <= qty_upper) & (df_clean["UnitPrice"] <= price_upper)
    ]

    print(
        f"Final cleaned dataset: {len(df_clean)} transactions for {df_clean['CustomerID'].nunique()} customers"
    )
    return df_clean


def calculate_rfm_metrics(df, analysis_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer

    Parameters:
    - df: Clean transaction DataFrame
    - analysis_date: Reference date for recency calculation (default: max date + 1 day)

    Returns:
    - DataFrame with RFM metrics per customer
    """
    print("Calculating RFM metrics...")

    if analysis_date is None:
        analysis_date = df["InvoiceDate"].max() + timedelta(days=1)

    # Calculate total spending per transaction
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    # Group by customer and calculate RFM metrics
    rfm = (
        df.groupby("CustomerID")
        .agg(
            {
                "InvoiceDate": lambda x: (analysis_date - x.max()).days,  # Recency
                "InvoiceNo": "count",  # Frequency
                "TotalAmount": "sum",  # Monetary
            }
        )
        .reset_index()
    )

    # Rename columns
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    print(f"RFM metrics calculated for {len(rfm)} customers")
    return rfm


def main():
    """
    Main function to execute the complete RFM analysis pipeline
    """
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PROJECT - RFM ANALYSIS")
    print("=" * 60)

    # Step 1: Create sample dataset
    df_raw = create_sample_dataset()

    # Step 2: Save raw dataset
    raw_data_path = "data/raw/sample_transactions.csv"
    df_raw.to_csv(raw_data_path, index=False)
    print(f"Raw dataset saved to: {raw_data_path}")

    # Step 3: Load dataset (demonstration)
    df_loaded = pd.read_csv(raw_data_path)
    df_loaded["InvoiceDate"] = pd.to_datetime(df_loaded["InvoiceDate"])
    print(f"Dataset loaded: {len(df_loaded)} transactions")

    # Step 4: Clean dataset
    df_clean = clean_dataset(df_loaded)

    # Step 5: Calculate RFM metrics
    rfm_table = calculate_rfm_metrics(df_clean)

    # Step 6: Save RFM table
    processed_data_path = "data/processed/rfm_table.csv"
    rfm_table.to_csv(processed_data_path, index=False)
    print(f"RFM table saved to: {processed_data_path}")

    # Step 7: Display first 10 customers
    print("\n" + "=" * 60)
    print("RFM TABLE - FIRST 10 CUSTOMERS")
    print("=" * 60)
    print(rfm_table.head(10).to_string(index=False))

    # Display summary statistics
    print("\n" + "=" * 60)
    print("RFM METRICS SUMMARY STATISTICS")
    print("=" * 60)
    print(rfm_table[["Recency", "Frequency", "Monetary"]].describe())

    print("\n" + "=" * 60)
    print("RFM ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
