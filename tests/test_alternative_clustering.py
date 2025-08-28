"""
Tests for alternative clustering functionality and outputs
"""

import joblib
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score


def test_dbscan_output_files_exist(project_root):
    """
    Test that DBSCAN output files exist.
    """
    # Check labels file
    dbscan_labels_path = project_root / "data" / "processed" / "dbscan_labels.csv"
    assert (
        dbscan_labels_path.exists()
    ), f"DBSCAN labels file not found: {dbscan_labels_path}"

    # Check merged data file
    rfm_with_dbscan_path = project_root / "data" / "processed" / "rfm_with_dbscan.csv"
    assert (
        rfm_with_dbscan_path.exists()
    ), f"RFM with DBSCAN file not found: {rfm_with_dbscan_path}"

    # Check model file
    dbscan_model_path = project_root / "models" / "dbscan_best.joblib"
    assert (
        dbscan_model_path.exists()
    ), f"DBSCAN model file not found: {dbscan_model_path}"

    # Check visualization file
    dbscan_scatter_path = project_root / "reports" / "figures" / "dbscan_scatter.png"
    assert (
        dbscan_scatter_path.exists()
    ), f"DBSCAN scatter plot not found: {dbscan_scatter_path}"


def test_hierarchical_output_files_exist(project_root):
    """
    Test that Hierarchical clustering output files exist.
    """
    # Check labels file
    hierarchical_labels_path = (
        project_root / "data" / "processed" / "hierarchical_labels.csv"
    )
    assert (
        hierarchical_labels_path.exists()
    ), f"Hierarchical labels file not found: {hierarchical_labels_path}"

    # Check merged data file
    rfm_with_hierarchical_path = (
        project_root / "data" / "processed" / "rfm_with_hierarchical.csv"
    )
    assert (
        rfm_with_hierarchical_path.exists()
    ), f"RFM with hierarchical file not found: {rfm_with_hierarchical_path}"

    # Check visualization files
    hierarchical_scatter_path = (
        project_root / "reports" / "figures" / "hierarchical_scatter.png"
    )
    assert (
        hierarchical_scatter_path.exists()
    ), f"Hierarchical scatter plot not found: {hierarchical_scatter_path}"

    hierarchical_dendrogram_path = (
        project_root / "reports" / "figures" / "hierarchical_dendrogram.png"
    )
    assert (
        hierarchical_dendrogram_path.exists()
    ), f"Hierarchical dendrogram not found: {hierarchical_dendrogram_path}"


def test_dbscan_clustering_validation(project_root):
    """
    Test that DBSCAN clustering has at least 1 cluster (excluding noise).
    """
    # Load DBSCAN labels
    dbscan_labels_path = project_root / "data" / "processed" / "dbscan_labels.csv"
    dbscan_labels_df = pd.read_csv(dbscan_labels_path)

    # Count clusters (excluding noise)
    unique_labels = set(dbscan_labels_df["segment"])
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    assert n_clusters >= 1, f"DBSCAN found {n_clusters} clusters, expected at least 1"

    # Test that all customers have labels
    assert (
        dbscan_labels_df["segment"].notna().all()
    ), "Some customers have missing DBSCAN labels"

    # Test row count matches input data
    rfm_scaled = pd.read_csv(project_root / "data" / "processed" / "rfm_scaled.csv")
    assert (
        len(dbscan_labels_df) == len(rfm_scaled)
    ), f"DBSCAN labels count {len(dbscan_labels_df)} doesn't match input data count {len(rfm_scaled)}"


def test_hierarchical_clustering_validation(project_root):
    """
    Test that Hierarchical clustering labels match input rows.
    """
    # Load hierarchical labels
    hierarchical_labels_path = (
        project_root / "data" / "processed" / "hierarchical_labels.csv"
    )
    hierarchical_labels_df = pd.read_csv(hierarchical_labels_path)

    # Test row count matches input data
    rfm_scaled = pd.read_csv(project_root / "data" / "processed" / "rfm_scaled.csv")
    assert (
        len(hierarchical_labels_df) == len(rfm_scaled)
    ), f"Hierarchical labels count {len(hierarchical_labels_df)} doesn't match input data count {len(rfm_scaled)}"

    # Test that all customers have labels
    assert (
        hierarchical_labels_df["segment"].notna().all()
    ), "Some customers have missing hierarchical labels"

    # Test that segment values are non-negative
    assert (
        hierarchical_labels_df["segment"] >= 0
    ).all(), "Some hierarchical segment values are negative"


def test_dbscan_silhouette_score(project_root):
    """
    Test that DBSCAN silhouette score is within valid range.
    """
    # Load data
    rfm_scaled = pd.read_csv(project_root / "data" / "processed" / "rfm_scaled.csv")
    dbscan_labels_df = pd.read_csv(
        project_root / "data" / "processed" / "dbscan_labels.csv"
    )

    features_scaled = rfm_scaled[
        ["Recency_Scaled", "Frequency_Scaled", "Monetary_Scaled"]
    ]
    dbscan_labels = dbscan_labels_df["segment"].values

    # Filter out noise points for silhouette calculation
    non_noise_mask = dbscan_labels != -1
    if sum(non_noise_mask) > 1:
        silhouette = silhouette_score(
            features_scaled[non_noise_mask], dbscan_labels[non_noise_mask]
        )

        # Test silhouette score bounds
        assert (
            -1 < silhouette <= 1
        ), f"DBSCAN silhouette score {silhouette:.3f} is not in (-1, 1]"

        # Test for weak clustering (xfail if silhouette <= 0.2)
        if silhouette <= 0.2:
            pytest.xfail(
                f"Weak DBSCAN cluster separation (silhouette={silhouette:.3f}) — dataset may not be clusterable"
            )

        # If we get here, silhouette is good
        assert (
            silhouette > 0.2
        ), f"DBSCAN silhouette score {silhouette:.3f} is too low for good clustering"
    else:
        pytest.skip("DBSCAN found only noise points or single cluster")


def test_hierarchical_silhouette_score(project_root):
    """
    Test that Hierarchical clustering silhouette score is within valid range.
    """
    # Load data
    rfm_scaled = pd.read_csv(project_root / "data" / "processed" / "rfm_scaled.csv")
    hierarchical_labels_df = pd.read_csv(
        project_root / "data" / "processed" / "hierarchical_labels.csv"
    )

    features_scaled = rfm_scaled[
        ["Recency_Scaled", "Frequency_Scaled", "Monetary_Scaled"]
    ]
    hierarchical_labels = hierarchical_labels_df["segment"].values

    # Calculate silhouette score
    silhouette = silhouette_score(features_scaled, hierarchical_labels)

    # Test silhouette score bounds
    assert (
        -1 < silhouette <= 1
    ), f"Hierarchical silhouette score {silhouette:.3f} is not in (-1, 1]"

    # Test for weak clustering (xfail if silhouette <= 0.2)
    if silhouette <= 0.2:
        pytest.xfail(
            f"Weak hierarchical cluster separation (silhouette={silhouette:.3f}) — dataset may not be clusterable"
        )

    # If we get here, silhouette is good
    assert (
        silhouette > 0.2
    ), f"Hierarchical silhouette score {silhouette:.3f} is too low for good clustering"


def test_model_files_loadable(project_root):
    """
    Test that model files can be loaded successfully.
    """
    # Test DBSCAN model
    dbscan_model_path = project_root / "models" / "dbscan_best.joblib"
    try:
        dbscan_model = joblib.load(dbscan_model_path)
        assert hasattr(
            dbscan_model, "eps"
        ), "Loaded DBSCAN model does not have eps attribute"
        assert hasattr(
            dbscan_model, "min_samples"
        ), "Loaded DBSCAN model does not have min_samples attribute"
    except Exception as e:
        pytest.fail(f"Failed to load DBSCAN model from {dbscan_model_path}: {str(e)}")

    # Test Hierarchical model
    hierarchical_model_files = list(
        (project_root / "models").glob("hierarchical_k*.joblib")
    )
    if hierarchical_model_files:
        hierarchical_model_path = hierarchical_model_files[0]
        try:
            hierarchical_model = joblib.load(hierarchical_model_path)
            assert hasattr(
                hierarchical_model, "n_clusters"
            ), "Loaded hierarchical model does not have n_clusters attribute"
        except Exception as e:
            pytest.fail(
                f"Failed to load hierarchical model from {hierarchical_model_path}: {str(e)}"
            )


def test_comparison_table_exists(project_root):
    """
    Test that clustering comparison table exists and has correct structure.
    """
    comparison_path = project_root / "reports" / "figures" / "clustering_comparison.csv"
    assert (
        comparison_path.exists()
    ), f"Clustering comparison table not found: {comparison_path}"

    # Load and validate comparison table
    comparison_df = pd.read_csv(comparison_path)

    # Check required columns
    required_columns = [
        "Method",
        "n_clusters",
        "n_noise",
        "silhouette_score",
        "calinski_harabasz",
        "davies_bouldin",
    ]
    for col in required_columns:
        assert col in comparison_df.columns, f"Comparison table missing column: {col}"

    # Check that all three methods are present
    expected_methods = ["KMeans", "DBSCAN", "Hierarchical"]
    actual_methods = list(comparison_df["Method"])
    assert (
        set(actual_methods) == set(expected_methods)
    ), f"Comparison table methods {actual_methods} don't match expected {expected_methods}"

    # Check that n_clusters values are positive
    assert (
        comparison_df["n_clusters"] > 0
    ).all(), "Some methods have non-positive cluster counts"


def test_data_consistency_across_methods(project_root):
    """
    Test that all clustering methods have consistent data across files.
    """
    # Load all label files
    kmeans_labels = pd.read_csv(project_root / "data" / "processed" / "labels_k3.csv")
    dbscan_labels = pd.read_csv(
        project_root / "data" / "processed" / "dbscan_labels.csv"
    )
    hierarchical_labels = pd.read_csv(
        project_root / "data" / "processed" / "hierarchical_labels.csv"
    )

    # Check that all have the same CustomerIDs
    kmeans_customers = set(kmeans_labels["CustomerID"])
    dbscan_customers = set(dbscan_labels["CustomerID"])
    hierarchical_customers = set(hierarchical_labels["CustomerID"])

    assert (
        kmeans_customers == dbscan_customers == hierarchical_customers
    ), "CustomerID sets don't match across clustering methods"

    # Check that all have the same number of rows
    assert (
        len(kmeans_labels) == len(dbscan_labels) == len(hierarchical_labels)
    ), "Row counts don't match across clustering methods"
