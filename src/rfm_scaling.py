"""
Customer Segmentation Project - Step 2: RFM Data Scaling
Scaling RFM metrics for clustering analysis using StandardScaler
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_rfm_data(file_path="data/processed/rfm_table.csv"):
    """
    Load the RFM table from the processed data folder
    
    Parameters:
    - file_path: Path to the RFM table CSV file
    
    Returns:
    - DataFrame with RFM metrics
    """
    print("Loading RFM data...")
    try:
        rfm_data = pd.read_csv(file_path)
        print(f"✓ Loaded RFM data: {len(rfm_data)} customers")
        print(f"  Columns: {list(rfm_data.columns)}")
        return rfm_data
    except FileNotFoundError:
        print(f"❌ Error: Could not find {file_path}")
        print("Please run the RFM analysis first (python src/rfm_analysis.py)")
        return None

def scale_rfm_data(rfm_data):
    """
    Scale RFM data using StandardScaler
    
    Parameters:
    - rfm_data: DataFrame with RFM metrics (CustomerID, Recency, Frequency, Monetary)
    
    Returns:
    - DataFrame with scaled RFM metrics
    - Fitted StandardScaler object
    """
    print("Scaling RFM data using StandardScaler...")
    
    # Separate features and customer IDs
    customer_ids = rfm_data['CustomerID']
    features = rfm_data[['Recency', 'Frequency', 'Monetary']]
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform the features
    features_scaled = scaler.fit_transform(features)
    
    # Create DataFrame with scaled features
    rfm_scaled = pd.DataFrame(
        features_scaled,
        columns=['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']
    )
    
    # Add customer IDs back
    rfm_scaled['CustomerID'] = customer_ids
    
    # Reorder columns to put CustomerID first
    rfm_scaled = rfm_scaled[['CustomerID', 'Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
    
    print("✓ RFM data scaled successfully")
    print(f"  Original features shape: {features.shape}")
    print(f"  Scaled features shape: {rfm_scaled.shape}")
    
    # Print scaling statistics
    print("\nScaling Statistics:")
    print(f"  Recency - Mean: {rfm_scaled['Recency_Scaled'].mean():.6f}, Std: {rfm_scaled['Recency_Scaled'].std():.6f}")
    print(f"  Frequency - Mean: {rfm_scaled['Frequency_Scaled'].mean():.6f}, Std: {rfm_scaled['Frequency_Scaled'].std():.6f}")
    print(f"  Monetary - Mean: {rfm_scaled['Monetary_Scaled'].mean():.6f}, Std: {rfm_scaled['Monetary_Scaled'].std():.6f}")
    
    return rfm_scaled, scaler

def save_scaled_data(rfm_scaled, file_path="data/processed/rfm_scaled.csv"):
    """
    Save the scaled RFM data to CSV file
    
    Parameters:
    - rfm_scaled: DataFrame with scaled RFM metrics
    - file_path: Path to save the scaled data
    """
    print(f"Saving scaled RFM data to {file_path}...")
    rfm_scaled.to_csv(file_path, index=False)
    print(f"✓ Scaled data saved successfully")

def display_scaled_data(rfm_scaled, n_rows=10):
    """
    Display the first n rows of scaled RFM data
    
    Parameters:
    - rfm_scaled: DataFrame with scaled RFM metrics
    - n_rows: Number of rows to display
    """
    print(f"\n{'='*60}")
    print(f"SCALED RFM DATA - FIRST {n_rows} CUSTOMERS")
    print(f"{'='*60}")
    print(rfm_scaled.head(n_rows).to_string(index=False))
    
    print(f"\n{'='*60}")
    print("SCALED DATA SUMMARY STATISTICS")
    print(f"{'='*60}")
    scaled_features = rfm_scaled[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
    print(scaled_features.describe())

def main():
    """
    Main function to execute the RFM scaling pipeline
    """
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PROJECT - STEP 2: RFM SCALING")
    print("=" * 60)
    
    # Step 1: Load RFM data
    rfm_data = load_rfm_data()
    if rfm_data is None:
        return
    
    # Step 2: Scale RFM data
    rfm_scaled, scaler = scale_rfm_data(rfm_data)
    
    # Step 3: Save scaled data
    save_scaled_data(rfm_scaled)
    
    # Step 4: Display results
    display_scaled_data(rfm_scaled)
    
    print("\n" + "=" * 60)
    print("RFM SCALING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Next steps:")
    print("1. Run clustering analysis on scaled data")
    print("2. Determine optimal number of clusters")
    print("3. Create customer segments")
    print("=" * 60)

if __name__ == "__main__":
    main()
