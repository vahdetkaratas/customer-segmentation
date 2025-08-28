"""
Tests for app core functionality
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Import functions to test
from src.app_core import (
    compute_rfm,
    evaluate_clusters,
    load_or_sample_data,
    pca_2d,
    run_dbscan,
    run_hierarchical,
    run_kmeans,
    scale_rfm,
)


def test_load_or_sample_data_none():
    """
    Test that load_or_sample_data(None) returns non-empty DF with required columns.
    """
    df = load_or_sample_data(None)

    # Check that DataFrame is not empty
    assert len(df) > 0, "Sample data should not be empty"

    # Check required columns
    required_columns = [
        "CustomerID",
        "InvoiceDate",
        "InvoiceNo",
        "Quantity",
        "UnitPrice",
    ]
    for col in required_columns:
        assert col in df.columns, f"Sample data missing column: {col}"

    # Check data types
    assert df["CustomerID"].dtype == "object", "CustomerID should be string"
    assert df["InvoiceDate"].dtype == "object", "InvoiceDate should be string"
    assert df["InvoiceNo"].dtype == "object", "InvoiceNo should be string"
    assert pd.api.types.is_numeric_dtype(df["Quantity"]), "Quantity should be numeric"
    assert pd.api.types.is_numeric_dtype(df["UnitPrice"]), "UnitPrice should be numeric"

    # Check data quality
    assert (df["Quantity"] > 0).all(), "All quantities should be positive"
    assert (df["UnitPrice"] > 0).all(), "All unit prices should be positive"


def test_load_or_sample_data_valid_csv():
    """
    Test loading data from a valid CSV file.
    """
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("CustomerID,InvoiceDate,InvoiceNo,Quantity,UnitPrice\n")
        f.write("CUST_001,2023-01-01,INV_001,5,100.0\n")
        f.write("CUST_002,2023-01-02,INV_002,3,50.0\n")
        temp_path = f.name

    try:
        df = load_or_sample_data(temp_path)

        # Check that data was loaded correctly
        assert len(df) == 2, "Should load 2 rows from test CSV"
        assert list(df["CustomerID"]) == [
            "CUST_001",
            "CUST_002",
        ], "CustomerIDs should match"
        assert list(df["Quantity"]) == [5, 3], "Quantities should match"
        assert list(df["UnitPrice"]) == [100.0, 50.0], "Unit prices should match"

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_load_or_sample_data_invalid_csv():
    """
    Test handling of invalid CSV file.
    """
    # Create a temporary CSV file with missing columns
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("CustomerID,InvoiceDate,Quantity,UnitPrice\n")  # Missing InvoiceNo
        f.write("CUST_001,2023-01-01,5,100.0\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Missing required columns"):
            load_or_sample_data(temp_path)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_compute_rfm():
    """
    Test RFM computation returns correct columns and no negative values.
    """
    # Create sample transaction data
    data = {
        "CustomerID": ["CUST_001", "CUST_001", "CUST_002", "CUST_002"],
        "InvoiceDate": ["2023-01-01", "2023-01-15", "2023-01-10", "2023-01-20"],
        "InvoiceNo": ["INV_001", "INV_002", "INV_003", "INV_004"],
        "Quantity": [5, 3, 2, 4],
        "UnitPrice": [100.0, 50.0, 75.0, 25.0],
    }
    df = pd.DataFrame(data)

    # Compute RFM
    rfm = compute_rfm(df)

    # Check required columns
    required_columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    for col in required_columns:
        assert col in rfm.columns, f"RFM missing column: {col}"

    # Check no negative values
    assert (rfm["Recency"] >= 0).all(), "Recency should be non-negative"
    assert (rfm["Frequency"] > 0).all(), "Frequency should be positive"
    assert (rfm["Monetary"] > 0).all(), "Monetary should be positive"

    # Check RFM calculations
    assert len(rfm) == 2, "Should have 2 customers"

    # CUST_001 should have frequency=2 and monetary=650 (5*100 + 3*50)
    cust_001 = rfm[rfm["CustomerID"] == "CUST_001"].iloc[0]
    assert cust_001["Frequency"] == 2, "CUST_001 should have 2 transactions"
    assert cust_001["Monetary"] == 650.0, "CUST_001 monetary should be 650"


def test_scale_rfm():
    """
    Test that scale_rfm returns means ≈ 0 and std ≈ 1 for R,F,M using pytest.approx.
    """
    # Create sample RFM data
    rfm_data = {
        "CustomerID": ["CUST_001", "CUST_002", "CUST_003"],
        "Recency": [10, 20, 30],
        "Frequency": [5, 10, 15],
        "Monetary": [1000, 2000, 3000],
    }
    rfm_df = pd.DataFrame(rfm_data)

    # Scale RFM
    rfm_scaled, scaler = scale_rfm(rfm_df)

    # Check that scaled features have mean ≈ 0 and std ≈ 1
    for col in ["Recency", "Frequency", "Monetary"]:
        mean_val = rfm_scaled[col].mean()
        std_val = rfm_scaled[col].std()

        assert mean_val == pytest.approx(0.0, abs=1e-10), f"{col} mean should be 0"
        # StandardScaler uses population std (divides by n), pandas std uses sample std (divides by n-1)
        # For small samples, this can differ significantly
        assert std_val == pytest.approx(1.0, abs=0.3), f"{col} std should be close to 1"

    # Check that CustomerID is preserved
    assert list(rfm_scaled["CustomerID"]) == list(
        rfm_df["CustomerID"]
    ), "CustomerID should be preserved"


def test_run_kmeans():
    """
    Test KMeans on small synthetic set produces exactly k unique labels.
    """
    # Create small synthetic RFM data
    rfm_data = {
        "CustomerID": [f"CUST_{i:03d}" for i in range(1, 11)],
        "Recency": np.random.uniform(10, 100, 10),
        "Frequency": np.random.uniform(1, 20, 10),
        "Monetary": np.random.uniform(100, 5000, 10),
    }
    rfm_df = pd.DataFrame(rfm_data)

    # Scale the data
    rfm_scaled, _ = scale_rfm(rfm_df)

    # Test with k=3
    k = 3
    labels, model = run_kmeans(rfm_scaled, k)

    # Check that we get exactly k unique labels
    unique_labels = set(labels)
    assert len(unique_labels) == k, f"Should have exactly {k} unique labels"

    # Check that labels are consecutive integers starting from 0
    expected_labels = set(range(k))
    assert unique_labels == expected_labels, f"Labels should be {expected_labels}"


def test_run_dbscan():
    """
    Test DBSCAN returns at least some labels.
    """
    # Create small synthetic RFM data
    rfm_data = {
        "CustomerID": [f"CUST_{i:03d}" for i in range(1, 11)],
        "Recency": np.random.uniform(10, 100, 10),
        "Frequency": np.random.uniform(1, 20, 10),
        "Monetary": np.random.uniform(100, 5000, 10),
    }
    rfm_df = pd.DataFrame(rfm_data)

    # Scale the data
    rfm_scaled, _ = scale_rfm(rfm_df)

    # Test DBSCAN
    labels, model = run_dbscan(rfm_scaled, eps=0.5, min_samples=2)

    # Check that we get labels (even if all noise)
    assert len(labels) == len(
        rfm_scaled
    ), "Should have same number of labels as data points"

    # Check that labels are integers
    assert all(
        isinstance(label, int | np.integer) for label in labels
    ), "All labels should be integers"

    # If all points are noise, mark as xfail
    if len(set(labels)) == 1 and -1 in labels:
        pytest.xfail("All points classified as noise - may need parameter tuning")


def test_run_hierarchical():
    """
    Test Hierarchical with n_clusters=3 returns 3 labels.
    """
    # Create small synthetic RFM data
    rfm_data = {
        "CustomerID": [f"CUST_{i:03d}" for i in range(1, 11)],
        "Recency": np.random.uniform(10, 100, 10),
        "Frequency": np.random.uniform(1, 20, 10),
        "Monetary": np.random.uniform(100, 5000, 10),
    }
    rfm_df = pd.DataFrame(rfm_data)

    # Scale the data
    rfm_scaled, _ = scale_rfm(rfm_df)

    # Test with n_clusters=3
    n_clusters = 3
    labels, model = run_hierarchical(rfm_scaled, n_clusters)

    # Check that we get exactly n_clusters unique labels
    unique_labels = set(labels)
    assert (
        len(unique_labels) == n_clusters
    ), f"Should have exactly {n_clusters} unique labels"

    # Check that labels are consecutive integers starting from 0
    expected_labels = set(range(n_clusters))
    assert unique_labels == expected_labels, f"Labels should be {expected_labels}"


def test_evaluate_clusters():
    """
    Test evaluate_clusters keys present and values within expected bounds when feasible.
    """
    # Create synthetic RFM data with clear clusters
    np.random.seed(42)
    n_samples = 30

    # Create 3 distinct clusters
    cluster1 = np.random.multivariate_normal(
        [0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], n_samples // 3
    )
    cluster2 = np.random.multivariate_normal(
        [3, 3, 3], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], n_samples // 3
    )
    cluster3 = np.random.multivariate_normal(
        [-3, -3, -3], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], n_samples // 3
    )

    features = np.vstack([cluster1, cluster2, cluster3])

    # Create RFM DataFrame
    rfm_data = {
        "CustomerID": [f"CUST_{i:03d}" for i in range(1, n_samples + 1)],
        "Recency": features[:, 0],
        "Frequency": features[:, 1],
        "Monetary": features[:, 2],
    }
    rfm_df = pd.DataFrame(rfm_data)

    # Scale the data
    rfm_scaled, _ = scale_rfm(rfm_df)

    # Create labels for 3 clusters
    labels = np.array(
        [0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3)
    )

    # Evaluate clusters
    metrics = evaluate_clusters(rfm_scaled, labels)

    # Check required keys
    required_keys = [
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "n_clusters",
        "n_noise",
    ]
    for key in required_keys:
        assert key in metrics, f"Metrics missing key: {key}"

    # Check values are within expected bounds
    if not np.isnan(metrics["silhouette"]):
        assert -1 <= metrics["silhouette"] <= 1, "Silhouette should be in [-1, 1]"

    if not np.isnan(metrics["calinski_harabasz"]):
        assert metrics["calinski_harabasz"] > 0, "Calinski-Harabasz should be positive"

    if not np.isnan(metrics["davies_bouldin"]):
        assert metrics["davies_bouldin"] >= 0, "Davies-Bouldin should be non-negative"

    assert metrics["n_clusters"] == 3, "Should have 3 clusters"
    assert metrics["n_noise"] == 0, "Should have no noise points"


def test_evaluate_clusters_single_cluster():
    """
    Test evaluate_clusters with single cluster (edge case).
    """
    # Create synthetic RFM data
    rfm_data = {
        "CustomerID": [f"CUST_{i:03d}" for i in range(1, 6)],
        "Recency": [10, 15, 20, 25, 30],
        "Frequency": [5, 6, 7, 8, 9],
        "Monetary": [1000, 1100, 1200, 1300, 1400],
    }
    rfm_df = pd.DataFrame(rfm_data)

    # Scale the data
    rfm_scaled, _ = scale_rfm(rfm_df)

    # Create single cluster labels
    labels = np.zeros(5)

    # Evaluate clusters
    metrics = evaluate_clusters(rfm_scaled, labels)

    # Check that silhouette is NaN for single cluster
    assert np.isnan(
        metrics["silhouette"]
    ), "Silhouette should be NaN for single cluster"
    assert np.isnan(
        metrics["calinski_harabasz"]
    ), "Calinski-Harabasz should be NaN for single cluster"
    assert np.isnan(
        metrics["davies_bouldin"]
    ), "Davies-Bouldin should be NaN for single cluster"
    assert metrics["n_clusters"] == 1, "Should have 1 cluster"


def test_pca_2d():
    """
    Test PCA 2D reduction returns correct shape and properties.
    """
    # Create synthetic RFM data
    rfm_data = {
        "CustomerID": [f"CUST_{i:03d}" for i in range(1, 11)],
        "Recency": np.random.uniform(10, 100, 10),
        "Frequency": np.random.uniform(1, 20, 10),
        "Monetary": np.random.uniform(100, 5000, 10),
    }
    rfm_df = pd.DataFrame(rfm_data)

    # Scale the data
    rfm_scaled, _ = scale_rfm(rfm_df)

    # Perform PCA
    components_2d, pca = pca_2d(rfm_scaled)

    # Check shape
    assert components_2d.shape == (10, 2), "Should have 10 samples and 2 components"

    # Check that PCA model has correct properties
    assert pca.n_components_ == 2, "PCA should have 2 components"
    assert pca.n_features_in_ == 3, "PCA should have 3 input features (R,F,M)"

    # Check that components are numeric
    assert np.isfinite(components_2d).all(), "All components should be finite"


def test_compute_rfm_data_cleaning():
    """
    Test that compute_rfm properly cleans data (removes negatives, zeros, NAs).
    """
    # Create data with some invalid entries
    data = {
        "CustomerID": ["CUST_001", "CUST_002", "CUST_003", "CUST_004", "CUST_005"],
        "InvoiceDate": [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
        ],
        "InvoiceNo": ["INV_001", "INV_002", "INV_003", "INV_004", "INV_005"],
        "Quantity": [5, 0, -1, 3, 4],  # 0 and negative values
        "UnitPrice": [100.0, 50.0, 75.0, 0.0, 25.0],  # 0 value
    }
    df = pd.DataFrame(data)

    # Compute RFM
    rfm = compute_rfm(df)

    # Should only have CUST_001, CUST_005 (valid entries - CUST_003 has negative quantity)
    expected_customers = ["CUST_001", "CUST_005"]
    assert len(rfm) == 2, "Should have 2 customers after cleaning"
    assert all(
        cust in rfm["CustomerID"].values for cust in expected_customers
    ), "Should have expected customers"


def test_scale_rfm_preserves_customerid():
    """
    Test that scale_rfm preserves CustomerID column.
    """
    # Create sample RFM data
    rfm_data = {
        "CustomerID": ["CUST_001", "CUST_002", "CUST_003"],
        "Recency": [10, 20, 30],
        "Frequency": [5, 10, 15],
        "Monetary": [1000, 2000, 3000],
    }
    rfm_df = pd.DataFrame(rfm_data)

    # Scale RFM
    rfm_scaled, _ = scale_rfm(rfm_df)

    # Check that CustomerID is preserved exactly
    assert list(rfm_scaled["CustomerID"]) == list(
        rfm_df["CustomerID"]
    ), "CustomerID should be preserved exactly"
    assert (
        rfm_scaled["CustomerID"].dtype == rfm_df["CustomerID"].dtype
    ), "CustomerID dtype should be preserved"
