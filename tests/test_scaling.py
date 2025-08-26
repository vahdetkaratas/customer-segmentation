"""
Tests for RFM data scaling functionality
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

def test_rfm_scaling_statistics(project_root):
    """
    Test that scaled RFM data has correct statistical properties.
    Each feature should have mean ≈ 0 and std ≈ 1.
    """
    # Load scaled data
    scaled_data_path = project_root / "data" / "processed" / "rfm_scaled.csv"
    
    if not scaled_data_path.exists():
        pytest.skip(f"Scaled data file not found: {scaled_data_path}")
    
    rfm_scaled = pd.read_csv(scaled_data_path)
    
    # Get scaled features (exclude CustomerID)
    scaled_features = rfm_scaled[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
    
    # Test each feature
    expected_mean = 0.0
    expected_std = 1.0
    tolerance = 1e-3
    
    for feature in scaled_features.columns:
        actual_mean = scaled_features[feature].mean()
        actual_std = scaled_features[feature].std()
        
        # Test mean
        assert actual_mean == pytest.approx(expected_mean, abs=tolerance), \
            f"{feature} mean {actual_mean:.8f} is not close to {expected_mean}"
        
        # Test standard deviation
        assert actual_std == pytest.approx(expected_std, abs=tolerance), \
            f"{feature} std {actual_std:.8f} is not close to {expected_std}"

def test_rfm_scaling_data_quality(project_root):
    """
    Test data quality of scaled RFM data.
    """
    # Load scaled data
    scaled_data_path = project_root / "data" / "processed" / "rfm_scaled.csv"
    
    if not scaled_data_path.exists():
        pytest.skip(f"Scaled data file not found: {scaled_data_path}")
    
    rfm_scaled = pd.read_csv(scaled_data_path)
    
    # Check for missing values
    missing_values = rfm_scaled.isnull().sum()
    assert missing_values.sum() == 0, f"Found {missing_values.sum()} missing values"
    
    # Check for infinite values
    scaled_features = rfm_scaled[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
    infinite_values = np.isinf(scaled_features).sum()
    assert infinite_values.sum() == 0, f"Found {infinite_values.sum()} infinite values"
    
    # Check data types
    expected_dtypes = {
        'CustomerID': 'object',
        'Recency_Scaled': 'float64',
        'Frequency_Scaled': 'float64',
        'Monetary_Scaled': 'float64'
    }
    
    for col, expected_dtype in expected_dtypes.items():
        assert str(rfm_scaled[col].dtype) == expected_dtype, \
            f"Column {col} has dtype {rfm_scaled[col].dtype}, expected {expected_dtype}"

def test_rfm_scaling_value_ranges(project_root):
    """
    Test that scaled values are within reasonable ranges.
    """
    # Load scaled data
    scaled_data_path = project_root / "data" / "processed" / "rfm_scaled.csv"
    
    if not scaled_data_path.exists():
        pytest.skip(f"Scaled data file not found: {scaled_data_path}")
    
    rfm_scaled = pd.read_csv(scaled_data_path)
    scaled_features = rfm_scaled[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
    
    # Check that values are within reasonable bounds (within 5 standard deviations)
    for feature in scaled_features.columns:
        min_val = scaled_features[feature].min()
        max_val = scaled_features[feature].max()
        
        assert min_val >= -5.0, f"{feature} has minimum value {min_val} < -5"
        assert max_val <= 5.0, f"{feature} has maximum value {max_val} > 5"

def test_rfm_scaling_row_count(project_root):
    """
    Test that scaled data has the same number of rows as original RFM data.
    """
    # Load both original and scaled data
    original_data_path = project_root / "data" / "processed" / "rfm_table.csv"
    scaled_data_path = project_root / "data" / "processed" / "rfm_scaled.csv"
    
    if not original_data_path.exists():
        pytest.skip(f"Original RFM data file not found: {original_data_path}")
    
    if not scaled_data_path.exists():
        pytest.skip(f"Scaled data file not found: {scaled_data_path}")
    
    rfm_original = pd.read_csv(original_data_path)
    rfm_scaled = pd.read_csv(scaled_data_path)
    
    assert len(rfm_original) == len(rfm_scaled), \
        f"Row count mismatch: original {len(rfm_original)} vs scaled {len(rfm_scaled)}"
    
    # Check that CustomerIDs match
    assert list(rfm_original['CustomerID']) == list(rfm_scaled['CustomerID']), \
        "CustomerID order or values do not match between original and scaled data"
