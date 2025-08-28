#!/usr/bin/env python3
"""
Tests for portfolio assets generation.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

# Add scripts to path
sys.path.append('scripts')

def test_sample_data_exists():
    """Test that sample datasets exist with correct structure."""
    # Test main sample dataset
    sample_path = 'data/sample/online_retail_sample.csv'
    assert os.path.exists(sample_path), f"Sample dataset not found: {sample_path}"
    
    df = pd.read_csv(sample_path)
    
    # Check required columns
    required_cols = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check data requirements
    assert len(df) >= 300, f"Expected >= 300 rows, got {len(df)}"
    assert df['CustomerID'].nunique() >= 120, f"Expected >= 120 unique customers, got {df['CustomerID'].nunique()}"
    
    # Check date span
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    date_span = (df['InvoiceDate'].max() - df['InvoiceDate'].min()).days
    assert date_span >= 60, f"Expected >= 60 days span, got {date_span}"
    
    # Check positive values
    assert (df['Quantity'] > 0).all(), "All quantities should be positive"
    assert (df['UnitPrice'] > 0).all(), "All unit prices should be positive"

def test_tiny_data_exists():
    """Test that tiny dataset exists with correct structure."""
    tiny_path = 'data/sample/online_retail_tiny.csv'
    assert os.path.exists(tiny_path), f"Tiny dataset not found: {tiny_path}"
    
    df = pd.read_csv(tiny_path)
    
    # Check required columns
    required_cols = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check size requirements
    assert 15 <= len(df) <= 25, f"Expected 15-25 rows, got {len(df)}"
    
    # Check positive values
    assert (df['Quantity'] > 0).all(), "All quantities should be positive"
    assert (df['UnitPrice'] > 0).all(), "All unit prices should be positive"

def test_asset_generation():
    """Test that portfolio assets are generated correctly."""
    # Import and run the asset generation script
    try:
        from make_portfolio_assets import main
        main()
    except ImportError:
        pytest.skip("Could not import make_portfolio_assets")
    
    # Check that figures exist
    figure_files = [
        'reports/figures/hero_pca_scatter.png',
        'reports/figures/hero_rfm_heatmap.png',
        'reports/figures/hero_segments_size.png',
        'reports/figures/hero_revenue_share.png'
    ]
    
    for file_path in figure_files:
        assert os.path.exists(file_path), f"Figure not generated: {file_path}"
        # Check file size > 0
        assert os.path.getsize(file_path) > 0, f"Figure file is empty: {file_path}"
    
    # Check segment profiles CSV
    profiles_path = 'reports/figures/portfolio_segment_profiles.csv'
    assert os.path.exists(profiles_path), f"Segment profiles not generated: {profiles_path}"
    
    profiles_df = pd.read_csv(profiles_path)
    
    # Check required columns
    required_cols = ['segment', 'n_customers', 'share_customers', 'revenue_share', 
                    'avg_recency', 'avg_frequency', 'avg_monetary']
    for col in required_cols:
        assert col in profiles_df.columns, f"Missing column in profiles: {col}"
    
    # Check that shares sum to approximately 1.0
    assert abs(profiles_df['share_customers'].sum() - 1.0) < 1e-6, "Customer shares don't sum to 1"
    assert abs(profiles_df['revenue_share'].sum() - 1.0) < 1e-6, "Revenue shares don't sum to 1"

@pytest.mark.xfail(reason="GIF skipped — imageio or sources missing")
def test_gif_generation():
    """Test that GIF is generated (xfail if imageio not available)."""
    gif_path = 'reports/figures/quicklook.gif'
    
    if os.path.exists(gif_path):
        assert os.path.getsize(gif_path) > 0, "GIF file is empty"
    else:
        pytest.xfail("GIF not generated - imageio or sources missing")

def test_readme_contains_step8():
    """Test that README contains Step 8 section."""
    readme_path = 'README.md'
    assert os.path.exists(readme_path), "README.md not found"
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for Step 8 section
    assert "Step 8: Portfolio Assets" in content or "Step 8: Portfolio Assets & Quick Demo" in content, \
        "README should contain Step 8 section"

def test_sample_data_structure():
    """Test that sample data has consistent structure."""
    # Load the sample data
    df = pd.read_csv('data/sample/online_retail_sample.csv')
    
    # Check that all required columns exist
    required_cols = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check that data types are consistent
    assert df['CustomerID'].dtype == 'object', "CustomerID should be string"
    assert df['InvoiceDate'].dtype == 'object', "InvoiceDate should be string"
    assert df['InvoiceNo'].dtype == 'object', "InvoiceNo should be string"
    assert df['Quantity'].dtype in ['int64', 'int32'], "Quantity should be integer"
    assert df['UnitPrice'].dtype in ['float64', 'float32'], "UnitPrice should be float"

def test_portfolio_assets_deterministic():
    """Test that portfolio assets generation is deterministic."""
    # Load the first set of assets
    profiles1 = pd.read_csv('reports/figures/portfolio_segment_profiles.csv')
    
    # Run the asset generation script again
    try:
        from make_portfolio_assets import main
        main()
    except ImportError:
        pytest.skip("Could not import make_portfolio_assets")
    
    # Load the assets again
    profiles2 = pd.read_csv('reports/figures/portfolio_segment_profiles.csv')
    
    # Check that the profiles are identical (within numerical precision)
    pd.testing.assert_frame_equal(profiles1, profiles2, check_dtype=False, rtol=1e-10), \
        "Portfolio assets generation is not deterministic"
