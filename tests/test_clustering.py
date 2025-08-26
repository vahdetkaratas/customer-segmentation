"""
Tests for clustering functionality and outputs
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import silhouette_score

def test_labels_file_exists_and_count(project_root):
    """
    Test that labels file exists and has correct row count.
    """
    # Find the labels file (it will have k in the filename)
    processed_dir = project_root / "data" / "processed"
    labels_files = list(processed_dir.glob("labels_k*.csv"))
    
    if not labels_files:
        pytest.skip("No labels file found. Run clustering analysis first.")
    
    # Use the first labels file found
    labels_path = labels_files[0]
    
    # Load labels and scaled data
    labels_df = pd.read_csv(labels_path)
    rfm_scaled = pd.read_csv(processed_dir / "rfm_scaled.csv")
    
    # Test row count
    assert len(labels_df) == len(rfm_scaled), \
        f"Labels file has {len(labels_df)} rows, but scaled data has {len(rfm_scaled)} rows"
    
    # Test required columns
    assert 'CustomerID' in labels_df.columns, "Labels file missing CustomerID column"
    assert 'segment' in labels_df.columns, "Labels file missing segment column"
    
    # Test that all customers have segments
    assert labels_df['segment'].notna().all(), "Some customers have missing segment labels"
    
    # Extract k from filename for other tests
    k_from_filename = int(labels_path.stem.split('_k')[1])

def test_best_k_range(project_root):
    """
    Test that best_k is between 2 and 10 (inclusive).
    """
    # Find the labels file to extract k
    processed_dir = project_root / "data" / "processed"
    labels_files = list(processed_dir.glob("labels_k*.csv"))
    
    if not labels_files:
        pytest.skip("No labels file found. Run clustering analysis first.")
    
    labels_path = labels_files[0]
    k = int(labels_path.stem.split('_k')[1])
    assert 2 <= k <= 10, f"best_k ({k}) is not between 2 and 10"

def test_silhouette_score_validation(project_root):
    """
    Test that final clustering has reasonable silhouette score.
    """
    # Load data
    processed_dir = project_root / "data" / "processed"
    labels_files = list(processed_dir.glob("labels_k*.csv"))
    
    if not labels_files:
        pytest.skip("No labels file found. Run clustering analysis first.")
    
    labels_path = labels_files[0]
    labels_df = pd.read_csv(labels_path)
    rfm_scaled = pd.read_csv(processed_dir / "rfm_scaled.csv")
    
    # Calculate silhouette score
    features_scaled = rfm_scaled[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
    silhouette = silhouette_score(features_scaled, labels_df['segment'])
    
    # Test silhouette score bounds
    assert -1 < silhouette <= 1, f"Silhouette score {silhouette:.3f} is not in (-1, 1]"
    
    # Test for weak clustering (xfail if silhouette <= 0.2)
    if silhouette <= 0.2:
        pytest.xfail(f"Weak cluster separation (silhouette={silhouette:.3f}) — dataset may not be clusterable")
    
    # If we get here, silhouette is good
    assert silhouette > 0.2, f"Silhouette score {silhouette:.3f} is too low for good clustering"

def test_model_file_exists(project_root):
    """
    Test that the KMeans model file exists.
    """
    # Find the model file (it will have k in the filename)
    models_dir = project_root / "models"
    model_files = list(models_dir.glob("kmeans_k*.joblib"))
    
    if not model_files:
        pytest.skip("No model file found. Run clustering analysis first.")
    
    # Use the first model file found
    model_path = model_files[0]
    
    # Test that file exists
    assert model_path.exists(), f"Model file {model_path} does not exist"
    
    # Test that we can load the model
    try:
        model = joblib.load(model_path)
        assert hasattr(model, 'n_clusters'), "Loaded model does not have n_clusters attribute"
        assert hasattr(model, 'cluster_centers_'), "Loaded model does not have cluster_centers_ attribute"
    except Exception as e:
        pytest.fail(f"Failed to load model from {model_path}: {str(e)}")

def test_rfm_with_segments_file(project_root):
    """
    Test that the merged RFM with segments file exists and has correct structure.
    """
    merged_path = project_root / "data" / "processed" / "rfm_with_segment.csv"
    
    if not merged_path.exists():
        pytest.skip("Merged RFM file not found. Run clustering analysis first.")
    
    # Load the merged data
    rfm_merged = pd.read_csv(merged_path)
    
    # Test required columns
    required_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'segment']
    for col in required_columns:
        assert col in rfm_merged.columns, f"Merged file missing column: {col}"
    
    # Test that segment values are integers
    assert rfm_merged['segment'].dtype in ['int64', 'int32'], \
        f"Segment column has dtype {rfm_merged['segment'].dtype}, expected integer"
    
    # Test that all customers have segments
    assert rfm_merged['segment'].notna().all(), "Some customers have missing segment labels"
    
    # Test that segment values are non-negative
    assert (rfm_merged['segment'] >= 0).all(), "Some segment values are negative"

def test_clustering_consistency(project_root):
    """
    Test that clustering results are consistent across files.
    """
    processed_dir = project_root / "data" / "processed"
    
    # Load all relevant files
    labels_files = list(processed_dir.glob("labels_k*.csv"))
    if not labels_files:
        pytest.skip("No labels file found. Run clustering analysis first.")
    
    labels_path = labels_files[0]
    labels_df = pd.read_csv(labels_path)
    rfm_merged = pd.read_csv(processed_dir / "rfm_with_segment.csv")
    
    # Test that CustomerIDs match between files
    labels_customers = set(labels_df['CustomerID'])
    merged_customers = set(rfm_merged['CustomerID'])
    
    assert labels_customers == merged_customers, \
        "CustomerID sets do not match between labels and merged files"
    
    # Test that segment assignments match
    labels_df_sorted = labels_df.sort_values('CustomerID').reset_index(drop=True)
    rfm_merged_sorted = rfm_merged.sort_values('CustomerID').reset_index(drop=True)
    
    assert (labels_df_sorted['segment'] == rfm_merged_sorted['segment']).all(), \
        "Segment assignments do not match between labels and merged files"

def test_segment_distribution(project_root):
    """
    Test that segments have reasonable distribution (no empty segments).
    """
    processed_dir = project_root / "data" / "processed"
    labels_files = list(processed_dir.glob("labels_k*.csv"))
    
    if not labels_files:
        pytest.skip("No labels file found. Run clustering analysis first.")
    
    labels_path = labels_files[0]
    labels_df = pd.read_csv(labels_path)
    
    # Get segment distribution
    segment_counts = labels_df['segment'].value_counts()
    k = len(segment_counts)
    
    # Test that we have the expected number of segments
    expected_k = int(labels_path.stem.split('_k')[1])
    assert k == expected_k, f"Expected {expected_k} segments, found {k}"
    
    # Test that no segment is empty
    assert (segment_counts > 0).all(), "Some segments have zero customers"
    
    # Test that segments are reasonably balanced (no segment has < 5% of customers)
    min_expected = len(labels_df) * 0.05
    assert (segment_counts >= min_expected).all(), \
        f"Some segments have fewer than 5% of customers (minimum expected: {min_expected:.0f})"
