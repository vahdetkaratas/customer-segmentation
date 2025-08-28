#!/usr/bin/env python3
"""
Core logic for Customer Segmentation Streamlit App
Reusable functions for data loading, RFM analysis, and clustering
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def load_or_sample_data(path: str | None = None) -> pd.DataFrame:
    """
    Load CSV data from path or generate reproducible sample data.

    Args:
        path: Path to CSV file. If None or file doesn't exist, generates sample data.

    Returns:
        DataFrame with columns: CustomerID, InvoiceDate, InvoiceNo, Quantity, UnitPrice
    """
    if path and Path(path).exists():
        try:
            df = pd.read_csv(path)
            required_cols = [
                "CustomerID",
                "InvoiceDate",
                "InvoiceNo",
                "Quantity",
                "UnitPrice",
            ]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")
    else:
        # Generate reproducible sample data
        np.random.seed(42)

        # Parameters for sample data
        n_customers = 200
        n_transactions = 1000
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)

        # Generate customer IDs
        customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]

        # Generate transaction data
        data = []
        for _ in range(n_transactions):
            customer_id = np.random.choice(customer_ids)
            invoice_date = start_date + timedelta(
                days=np.random.randint(0, (end_date - start_date).days)
            )
            invoice_no = f"INV_{np.random.randint(10000, 99999)}"
            quantity = np.random.randint(1, 50)
            unit_price = np.random.uniform(10, 1000)

            data.append(
                {
                    "CustomerID": customer_id,
                    "InvoiceDate": invoice_date.strftime("%Y-%m-%d"),
                    "InvoiceNo": invoice_no,
                    "Quantity": quantity,
                    "UnitPrice": unit_price,
                }
            )

        df = pd.DataFrame(data)
        return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM metrics from transaction data.

    Args:
        df: DataFrame with columns: CustomerID, InvoiceDate, InvoiceNo, Quantity, UnitPrice

    Returns:
        DataFrame with columns: CustomerID, Recency, Frequency, Monetary
    """
    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Calculate total amount per transaction
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    # Clean data: remove negatives and zeros
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0) & (df["TotalAmount"] > 0)]

    # Drop NAs
    df = df.dropna()

    # Calculate RFM metrics
    analysis_date = df["InvoiceDate"].max()

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

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    # Ensure no negative values
    rfm = rfm[(rfm["Recency"] >= 0) & (rfm["Frequency"] > 0) & (rfm["Monetary"] > 0)]

    return rfm


def scale_rfm(rfm_df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scale RFM features using StandardScaler.

    Args:
        rfm_df: DataFrame with Recency, Frequency, Monetary columns

    Returns:
        Tuple of (scaled DataFrame, fitted StandardScaler)
    """
    # Extract features for scaling
    features = rfm_df[["Recency", "Frequency", "Monetary"]].copy()

    # Fit and transform
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create scaled DataFrame
    rfm_scaled = rfm_df.copy()
    rfm_scaled[["Recency", "Frequency", "Monetary"]] = features_scaled

    return rfm_scaled, scaler


def run_kmeans(
    rfm_scaled: pd.DataFrame, k: int, random_state: int = 42
) -> tuple[np.ndarray, KMeans]:
    """
    Run KMeans clustering on scaled RFM data.

    Args:
        rfm_scaled: Scaled RFM DataFrame
        k: Number of clusters
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (cluster labels, fitted KMeans model)
    """
    features = rfm_scaled[["Recency", "Frequency", "Monetary"]].values

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)

    return labels, kmeans


def run_dbscan(
    rfm_scaled: pd.DataFrame, eps: float, min_samples: int
) -> tuple[np.ndarray, DBSCAN]:
    """
    Run DBSCAN clustering on scaled RFM data.

    Args:
        rfm_scaled: Scaled RFM DataFrame
        eps: Epsilon parameter for DBSCAN
        min_samples: Minimum samples parameter for DBSCAN

    Returns:
        Tuple of (cluster labels, fitted DBSCAN model)
    """
    features = rfm_scaled[["Recency", "Frequency", "Monetary"]].values

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)

    return labels, dbscan


def run_hierarchical(
    rfm_scaled: pd.DataFrame, n_clusters: int
) -> tuple[np.ndarray, AgglomerativeClustering]:
    """
    Run Hierarchical clustering on scaled RFM data.

    Args:
        rfm_scaled: Scaled RFM DataFrame
        n_clusters: Number of clusters

    Returns:
        Tuple of (cluster labels, fitted AgglomerativeClustering model)
    """
    features = rfm_scaled[["Recency", "Frequency", "Monetary"]].values

    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = hierarchical.fit_predict(features)

    return labels, hierarchical


def evaluate_clusters(rfm_scaled: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """
    Evaluate clustering results using multiple metrics.

    Args:
        rfm_scaled: Scaled RFM DataFrame
        labels: Cluster labels

    Returns:
        Dictionary with evaluation metrics
    """
    features = rfm_scaled[["Recency", "Frequency", "Monetary"]].values
    n_clusters = len(set(labels)) - (
        1 if -1 in labels else 0
    )  # Exclude noise for DBSCAN

    metrics = {}

    # Silhouette Score (requires at least 2 clusters)
    if n_clusters >= 2:
        try:
            # For DBSCAN, exclude noise points
            if -1 in labels:
                non_noise_mask = labels != -1
                if sum(non_noise_mask) > 1:
                    metrics["silhouette"] = silhouette_score(
                        features[non_noise_mask], labels[non_noise_mask]
                    )
                else:
                    metrics["silhouette"] = np.nan
            else:
                metrics["silhouette"] = silhouette_score(features, labels)
        except:
            metrics["silhouette"] = np.nan
    else:
        metrics["silhouette"] = np.nan

    # Calinski-Harabasz Index (requires at least 2 clusters)
    if n_clusters >= 2:
        try:
            if -1 in labels:
                non_noise_mask = labels != -1
                if sum(non_noise_mask) > 1:
                    metrics["calinski_harabasz"] = calinski_harabasz_score(
                        features[non_noise_mask], labels[non_noise_mask]
                    )
                else:
                    metrics["calinski_harabasz"] = np.nan
            else:
                metrics["calinski_harabasz"] = calinski_harabasz_score(features, labels)
        except:
            metrics["calinski_harabasz"] = np.nan
    else:
        metrics["calinski_harabasz"] = np.nan

    # Davies-Bouldin Index (requires at least 2 clusters)
    if n_clusters >= 2:
        try:
            if -1 in labels:
                non_noise_mask = labels != -1
                if sum(non_noise_mask) > 1:
                    metrics["davies_bouldin"] = davies_bouldin_score(
                        features[non_noise_mask], labels[non_noise_mask]
                    )
                else:
                    metrics["davies_bouldin"] = np.nan
            else:
                metrics["davies_bouldin"] = davies_bouldin_score(features, labels)
        except:
            metrics["davies_bouldin"] = np.nan
    else:
        metrics["davies_bouldin"] = np.nan

    # Add cluster count
    metrics["n_clusters"] = n_clusters
    metrics["n_noise"] = sum(labels == -1) if -1 in labels else 0

    return metrics


def pca_2d(rfm_scaled: pd.DataFrame) -> tuple[np.ndarray, PCA]:
    """
    Perform PCA to reduce RFM data to 2D for visualization.

    Args:
        rfm_scaled: Scaled RFM DataFrame

    Returns:
        Tuple of (2D components, fitted PCA model)
    """
    features = rfm_scaled[["Recency", "Frequency", "Monetary"]].values

    pca = PCA(n_components=2, random_state=42)
    components_2d = pca.fit_transform(features)

    return components_2d, pca


def save_app_artifacts(
    rfm_df: pd.DataFrame,
    rfm_scaled: pd.DataFrame,
    labels: np.ndarray,
    customer_ids: pd.Series,
) -> None:
    """
    Save minimal artifacts for the app.

    Args:
        rfm_df: Original RFM DataFrame
        rfm_scaled: Scaled RFM DataFrame
        labels: Cluster labels
        customer_ids: Customer IDs
    """
    # Ensure directory exists
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Save RFM table
    rfm_df.to_csv("data/processed/rfm_table.csv", index=False)

    # Save scaled RFM
    rfm_scaled.to_csv("data/processed/rfm_scaled.csv", index=False)

    # Save segments
    segments_df = pd.DataFrame(
        {
            "CustomerID": customer_ids,
            "segment": labels,
            "Recency": rfm_df["Recency"],
            "Frequency": rfm_df["Frequency"],
            "Monetary": rfm_df["Monetary"],
        }
    )
    segments_df.to_csv("data/processed/segments_app.csv", index=False)
