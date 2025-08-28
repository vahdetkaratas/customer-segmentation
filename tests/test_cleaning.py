#!/usr/bin/env python3
"""
Tests for data cleaning functionality.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from cleaning import (
    set_seed, ensure_dirs, load_config, load_transactions,
    validate_schema, coerce_types, analyze_issues, propose_plan,
    apply_plan, detect_outliers_iqr, winsorize_series
)


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    
    # Create realistic sample data
    n_rows = 50
    customers = [f"CUST{i:03d}" for i in range(1, 11)]
    dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
    
    data = {
        'CustomerID': np.random.choice(customers, n_rows),
        'InvoiceDate': np.random.choice(dates, n_rows),
        'InvoiceNo': [f"INV{i:06d}" for i in range(1, n_rows + 1)],
        'Quantity': np.random.randint(1, 100, n_rows),
        'UnitPrice': np.random.uniform(1.0, 100.0, n_rows)
    }
    
    # Add some problematic data
    data['Quantity'][0] = -5  # Invalid quantity
    data['UnitPrice'][1] = 0  # Invalid price
    data['Quantity'][2] = 1000  # Outlier
    data['UnitPrice'][3] = 500  # Outlier
    
    return pd.DataFrame(data)


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'expected_columns': ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice'],
        'date': {
            'required': True,
            'min_span_days': 30,
            'formats_hint': ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S']
        },
        'duplicates': {
            'drop_exact': True,
            'consider_near_duplicates': True,
            'near_keys': ['CustomerID', 'InvoiceNo', 'InvoiceDate']
        },
        'values': {
            'quantity_min': 1,
            'unitprice_min': 0.01
        },
        'outliers': {
            'method': 'iqr',
            'iqr_multiplier': 1.5,
            'handle': 'winsorize',
            'winsor_limits': [0.01, 0.99]
        },
        'customers': {
            'flag_min_invoices': 2,
            'drop_thin_customers': False
        },
        'outputs': {
            'dryrun_report_md': 'reports/cleaning_report.md',
            'dryrun_report_json': 'reports/cleaning_findings.json',
            'cleaned_csv': 'data/processed/transactions_cleaned.csv',
            'rejected_csv': 'data/processed/transactions_rejected.csv',
            'issues_csv': 'data/processed/transactions_issues_catalog.csv'
        }
    }


def test_set_seed():
    """Test random seed setting."""
    set_seed(42)
    # Should not raise any exceptions
    assert True


def test_ensure_dirs(tmp_path):
    """Test directory creation."""
    test_file = tmp_path / "subdir" / "test.txt"
    ensure_dirs(str(test_file))
    assert test_file.parent.exists()


def test_load_config():
    """Test config loading."""
    config = load_config("config/cleaning_rules.yml")
    assert 'expected_columns' in config
    assert 'outliers' in config
    assert config['expected_columns'] == ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice']


def test_load_transactions():
    """Test transaction loading."""
    # Test with existing sample file
    if os.path.exists('data/sample/online_retail_tiny.csv'):
        df = load_transactions('data/sample/online_retail_tiny.csv')
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'CustomerID' in df.columns


def test_validate_schema(sample_data, test_config):
    """Test schema validation."""
    issues = validate_schema(sample_data, test_config['expected_columns'])
    
    assert 'missing_columns' in issues
    assert 'extra_columns' in issues
    assert 'type_issues' in issues
    assert len(issues['missing_columns']) == 0  # All expected columns present
    
    # Test with missing column
    df_missing = sample_data.drop(columns=['Quantity'])
    issues_missing = validate_schema(df_missing, test_config['expected_columns'])
    assert 'Quantity' in issues_missing['missing_columns']


def test_coerce_types(sample_data):
    """Test type coercion."""
    df_clean, issues = coerce_types(sample_data)
    
    assert isinstance(df_clean, pd.DataFrame)
    assert 'date_parse_failures' in issues
    assert 'numeric_parse_failures' in issues
    assert 'string_issues' in issues
    
    # Check that dates are parsed
    assert pd.api.types.is_datetime64_any_dtype(df_clean['InvoiceDate'])
    
    # Check that numeric columns are numeric
    assert pd.api.types.is_numeric_dtype(df_clean['Quantity'])
    assert pd.api.types.is_numeric_dtype(df_clean['UnitPrice'])


def test_detect_outliers_iqr():
    """Test outlier detection."""
    # Create series with known outliers
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
    series = pd.Series(data)
    
    outliers = detect_outliers_iqr(series, k=1.5)
    assert outliers.sum() > 0
    assert outliers.iloc[-1] == True  # Last value (100) should be outlier


def test_winsorize_series():
    """Test winsorization."""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    series = pd.Series(data)
    
    winsorized = winsorize_series(series, lower=0.1, upper=0.9)
    assert len(winsorized) == len(series)
    assert winsorized.max() < series.max()  # Should cap the maximum


def test_analyze_issues(sample_data, test_config):
    """Test issue analysis."""
    issues = analyze_issues(sample_data, test_config)
    
    assert 'total_rows' in issues
    assert 'invalid_values' in issues
    assert 'outliers' in issues
    assert 'date_span' in issues
    assert 'thin_customers' in issues
    assert 'duplicates' in issues
    
    # Should detect invalid values
    assert issues['invalid_values'].get('Quantity', 0) > 0
    assert issues['invalid_values'].get('UnitPrice', 0) > 0
    
    # Should detect outliers
    assert issues['outliers'].get('Quantity', 0) > 0 or issues['outliers'].get('UnitPrice', 0) > 0


def test_propose_plan(sample_data, test_config):
    """Test plan proposal."""
    issues = analyze_issues(sample_data, test_config)
    plan = propose_plan(issues, test_config)
    
    assert isinstance(plan, list)
    assert len(plan) > 0  # Should propose at least one step
    
    # Check plan structure
    for step in plan:
        assert 'step' in step
        assert 'target' in step
        assert 'reason' in step
        assert 'count_estimate' in step


def test_apply_plan(sample_data, test_config):
    """Test plan application."""
    issues = analyze_issues(sample_data, test_config)
    plan = propose_plan(issues, test_config)
    
    cleaned_df, rejected_df, issues_df = apply_plan(sample_data, plan, test_config)
    
    assert isinstance(cleaned_df, pd.DataFrame)
    assert isinstance(rejected_df, pd.DataFrame)
    assert isinstance(issues_df, pd.DataFrame)
    
    # Check that cleaned data has no invalid values
    if len(cleaned_df) > 0:
        invalid_qty = (cleaned_df['Quantity'] <= 0).sum()
        invalid_price = (cleaned_df['UnitPrice'] <= 0).sum()
        assert invalid_qty == 0
        assert invalid_price == 0


def test_analyze_path_dry_run():
    """Test analyze path (dry-run) using CLI."""
    if not os.path.exists('data/sample/online_retail_tiny.csv'):
        pytest.skip("Sample data not available")
    
    # Run analyze mode
    import subprocess
    result = subprocess.run([
        'python', 'scripts/run_cleaning.py',
        '--input', 'data/sample/online_retail_tiny.csv'
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    
    # Check that reports were created
    assert os.path.exists('reports/cleaning_report.md')
    assert os.path.exists('reports/cleaning_findings.json')


def test_apply_path():
    """Test apply path using CLI."""
    if not os.path.exists('data/sample/online_retail_tiny.csv'):
        pytest.skip("Sample data not available")
    
    # Run apply mode
    import subprocess
    result = subprocess.run([
        'python', 'scripts/run_cleaning.py',
        '--input', 'data/sample/online_retail_tiny.csv',
        '--apply'
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    
    # Check that output files were created
    assert os.path.exists('data/processed/transactions_cleaned.csv')
    assert os.path.exists('data/processed/transactions_rejected.csv')
    assert os.path.exists('data/processed/transactions_issues_catalog.csv')
    
    # Check that cleaned data has rows
    cleaned_df = pd.read_csv('data/processed/transactions_cleaned.csv')
    assert len(cleaned_df) > 0
    
    # Check that cleaned data has no invalid values
    if 'Quantity' in cleaned_df.columns and 'UnitPrice' in cleaned_df.columns:
        invalid_qty = (cleaned_df['Quantity'] <= 0).sum()
        invalid_price = (cleaned_df['UnitPrice'] <= 0).sum()
        assert invalid_qty == 0
        assert invalid_price == 0


def test_outlier_handling():
    """Test outlier detection and handling."""
    # Create data with extreme outliers
    data = pd.DataFrame({
        'CustomerID': ['CUST001'] * 10,
        'InvoiceDate': pd.date_range('2023-01-01', periods=10),
        'InvoiceNo': [f'INV{i:03d}' for i in range(1, 11)],
        'Quantity': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10000],  # Extreme outlier
        'UnitPrice': [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]  # Extreme outlier
    })
    
    config = {
        'expected_columns': ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice'],
        'date': {
            'required': True,
            'min_span_days': 30,
            'formats_hint': ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S']
        },
        'outliers': {
            'method': 'iqr',
            'iqr_multiplier': 1.5,
            'handle': 'winsorize',
            'winsor_limits': [0.01, 0.99]
        },
        'duplicates': {'drop_exact': True, 'near_keys': ['CustomerID', 'InvoiceNo', 'InvoiceDate']},
        'customers': {'flag_min_invoices': 2, 'drop_thin_customers': False}
    }
    
    issues = analyze_issues(data, config)
    plan = propose_plan(issues, config)
    
    # Should detect outliers
    assert issues['outliers'].get('Quantity', 0) > 0
    assert issues['outliers'].get('UnitPrice', 0) > 0
    
    # Should propose winsorize step
    winsorize_steps = [step for step in plan if step['step'] == 'winsorize_outliers']
    assert len(winsorize_steps) > 0
    
    # Test winsorization
    cleaned_df, _, _ = apply_plan(data, plan, config)
    assert len(cleaned_df) == len(data)  # Should keep all rows
    assert cleaned_df['Quantity'].max() < data['Quantity'].max()  # Should cap outliers


def test_sample_data_structure():
    """Test that sample data has consistent structure."""
    if not os.path.exists('data/sample/online_retail_tiny.csv'):
        pytest.skip("Sample data not available")
    
    df = pd.read_csv('data/sample/online_retail_tiny.csv')
    
    required_cols = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    assert df['CustomerID'].dtype == 'object', "CustomerID should be string"
    assert df['InvoiceDate'].dtype == 'object', "InvoiceDate should be string"
    assert df['InvoiceNo'].dtype == 'object', "InvoiceNo should be string"
    assert df['Quantity'].dtype in ['int64', 'int32'], "Quantity should be integer"
    assert df['UnitPrice'].dtype in ['float64', 'float32'], "UnitPrice should be float"


@pytest.mark.xfail
def test_gif_generation():
    """Test GIF generation (marked as xfail if imageio not available)."""
    # This test would check if quicklook.gif was created
    # Marked as xfail since imageio might not be available
    assert os.path.exists('reports/figures/quicklook.gif')
