#!/usr/bin/env python3
"""
Customer Segmentation Streamlit App
Interactive web application for RFM analysis and clustering
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from app_core import (
    load_or_sample_data, compute_rfm, scale_rfm, run_kmeans, 
    run_dbscan, run_hierarchical, evaluate_clusters, pca_2d, save_app_artifacts
)

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file=None, use_sample=False):
    """Load data with caching"""
    if use_sample or uploaded_file is None:
        return load_or_sample_data()
    else:
        # Save uploaded file temporarily
        temp_path = "temp_upload.csv"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            data = load_or_sample_data(temp_path)
            # Clean up temp file
            if Path(temp_path).exists():
                Path(temp_path).unlink()
            return data
        except Exception as e:
            # Clean up temp file
            if Path(temp_path).exists():
                Path(temp_path).unlink()
            raise e

@st.cache_data
def process_rfm(data):
    """Process RFM with caching"""
    rfm = compute_rfm(data)
    rfm_scaled, scaler = scale_rfm(rfm)
    return rfm, rfm_scaled, scaler

@st.cache_data
def run_clustering_algorithm(rfm_scaled, algorithm, **params):
    """Run clustering with caching"""
    if algorithm == "KMeans":
        return run_kmeans(rfm_scaled, params['k'])
    elif algorithm == "DBSCAN":
        return run_dbscan(rfm_scaled, params['eps'], params['min_samples'])
    elif algorithm == "Hierarchical":
        return run_hierarchical(rfm_scaled, params['n_clusters'])

def create_pca_plot(rfm_scaled, labels, algorithm):
    """Create PCA scatter plot"""
    components_2d, pca = pca_2d(rfm_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique labels (excluding noise for DBSCAN)
    unique_labels = sorted(set(labels))
    
    # Create color map
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        if label == -1:  # Noise points for DBSCAN
            mask = labels == label
            ax.scatter(components_2d[mask, 0], components_2d[mask, 1], 
                      c='red', marker='x', s=100, alpha=0.7, 
                      label=f'Noise (n={sum(mask)})')
        else:
            mask = labels == label
            ax.scatter(components_2d[mask, 0], components_2d[mask, 1], 
                      c=[colors[i]], s=50, alpha=0.7, 
                      label=f'Cluster {label} (n={sum(mask)})')
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title(f'PCA 2D Visualization - {algorithm} Clustering')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def create_heatmap(rfm_df, labels):
    """Create segment profile heatmap"""
    # Add labels to RFM data
    rfm_with_labels = rfm_df.copy()
    rfm_with_labels['segment'] = labels
    
    # Calculate mean RFM per segment
    segment_profiles = rfm_with_labels.groupby('segment')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Normalize for heatmap (column-wise min-max scaling)
    segment_profiles_norm = (segment_profiles - segment_profiles.min()) / (segment_profiles.max() - segment_profiles.min())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(segment_profiles_norm.T, cmap='RdYlBu_r', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(len(segment_profiles)))
    ax.set_xticklabels([f'Segment {seg}' for seg in segment_profiles.index])
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Recency', 'Frequency', 'Monetary'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Value')
    
    # Add text annotations
    for i in range(len(segment_profiles)):
        for j in range(3):
            text = ax.text(i, j, f'{segment_profiles.iloc[i, j]:.1f}', 
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Segment RFM Profile Heatmap')
    
    return fig

def main():
    """Main Streamlit app"""
    # Header
    st.markdown('<h1 class="main-header">Customer Segmentation App</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        help="Upload a CSV file with columns: CustomerID, InvoiceDate, InvoiceNo, Quantity, UnitPrice"
    )
    
    # Sample data option
    use_sample = st.sidebar.checkbox("Use sample data instead", value=True)
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Clustering Algorithm",
        ["KMeans", "DBSCAN", "Hierarchical"]
    )
    
    # Algorithm-specific parameters
    if algorithm == "KMeans":
        k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
        params = {'k': k}
    elif algorithm == "DBSCAN":
        eps = st.sidebar.slider("Epsilon (eps)", 0.1, 1.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("Min samples", 2, 20, 5)
        params = {'eps': eps, 'min_samples': min_samples}
    else:  # Hierarchical
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
        params = {'n_clusters': n_clusters}
    
    # Run button
    run_button = st.sidebar.button("🚀 Run Clustering", type="primary")
    
    # Main area
    if not run_button:
        # Show instructions
        st.markdown("""
        <div class="info-box">
        <h3>📋 Instructions</h3>
        <p>This app performs customer segmentation using RFM (Recency, Frequency, Monetary) analysis and clustering algorithms.</p>
        
        <h4>📁 Data Requirements</h4>
        <p>Your CSV file should contain these columns:</p>
        <ul>
            <li><strong>CustomerID</strong>: Unique customer identifier</li>
            <li><strong>InvoiceDate</strong>: Date of transaction (YYYY-MM-DD format)</li>
            <li><strong>InvoiceNo</strong>: Invoice number</li>
            <li><strong>Quantity</strong>: Number of items purchased</li>
            <li><strong>UnitPrice</strong>: Price per unit</li>
        </ul>
        
        <h4>🔧 How to Use</h4>
        <ol>
            <li>Upload your CSV file or check "Use sample data"</li>
            <li>Select a clustering algorithm and set parameters</li>
            <li>Click "Run Clustering" to analyze your data</li>
            <li>Review results and download segments</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample data preview
        if use_sample:
            st.subheader("📊 Sample Data Preview")
            sample_data = load_or_sample_data()
            st.dataframe(sample_data.head(10), use_container_width=True)
            st.caption(f"Sample data contains {len(sample_data)} transactions from {sample_data['CustomerID'].nunique()} customers")
        
        return
    
    # Run clustering analysis
    try:
        with st.spinner("Loading data..."):
            data = load_data(uploaded_file, use_sample)
        
        with st.spinner("Computing RFM metrics..."):
            rfm, rfm_scaled, scaler = process_rfm(data)
        
        with st.spinner(f"Running {algorithm} clustering..."):
            labels, model = run_clustering_algorithm(rfm_scaled, algorithm, **params)
        
        with st.spinner("Evaluating results..."):
            metrics = evaluate_clusters(rfm_scaled, labels)
        
        # Save artifacts
        save_app_artifacts(rfm, rfm_scaled, labels, rfm['CustomerID'])
        
        # Display results
        st.success("✅ Clustering completed successfully!")
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", len(rfm))
        with col2:
            st.metric("Total Transactions", len(data))
        with col3:
            st.metric("Clusters Found", metrics['n_clusters'])
        with col4:
            if algorithm == "DBSCAN":
                st.metric("Noise Points", metrics['n_noise'])
            else:
                st.metric("Avg Recency (days)", f"{rfm['Recency'].mean():.1f}")
        
        # RFM Summary
        st.subheader("📈 RFM Summary")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(rfm.head(10), use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h4>RFM Metrics</h4>
            <p><strong>Recency:</strong> Days since last purchase</p>
            <p><strong>Frequency:</strong> Number of transactions</p>
            <p><strong>Monetary:</strong> Total spending amount</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Clustering Metrics
        st.subheader("📊 Clustering Evaluation")
        
        if not np.isnan(metrics['silhouette']):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}")
            with col2:
                if not np.isnan(metrics['calinski_harabasz']):
                    st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:.1f}")
                else:
                    st.metric("Calinski-Harabasz", "N/A")
            with col3:
                if not np.isnan(metrics['davies_bouldin']):
                    st.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}")
                else:
                    st.metric("Davies-Bouldin", "N/A")
        else:
            st.warning("⚠️ Silhouette score not available (requires at least 2 clusters)")
        
        # Visualizations
        st.subheader("📊 Visualizations")
        
        # PCA Plot
        st.write("**PCA 2D Scatter Plot**")
        pca_fig = create_pca_plot(rfm_scaled, labels, algorithm)
        st.pyplot(pca_fig)
        plt.close()
        
        # Heatmap
        st.write("**Segment RFM Profile Heatmap**")
        heatmap_fig = create_heatmap(rfm, labels)
        st.pyplot(heatmap_fig)
        plt.close()
        
        # Download section
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download segments
            segments_df = pd.DataFrame({
                'CustomerID': rfm['CustomerID'],
                'segment': labels,
                'Recency': rfm['Recency'],
                'Frequency': rfm['Frequency'],
                'Monetary': rfm['Monetary']
            })
            
            csv_segments = segments_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Segments CSV",
                data=csv_segments,
                file_name=f"customer_segments_{algorithm.lower()}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download RFM data
            csv_rfm = rfm.to_csv(index=False)
            st.download_button(
                label="📥 Download RFM CSV",
                data=csv_rfm,
                file_name="rfm_metrics.csv",
                mime="text/csv"
            )
        
        # Algorithm-specific notes
        if algorithm == "DBSCAN":
            st.info("""
            **DBSCAN Notes:**
            - Points labeled as -1 are noise points (outliers)
            - Silhouette score excludes noise points
            - Adjust eps and min_samples for different cluster densities
            """)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.markdown("""
        <div class="info-box">
        <h4>🔧 Troubleshooting</h4>
        <p>Common issues:</p>
        <ul>
            <li><strong>Missing columns:</strong> Ensure your CSV has CustomerID, InvoiceDate, InvoiceNo, Quantity, UnitPrice</li>
            <li><strong>Date format:</strong> Use YYYY-MM-DD format for InvoiceDate</li>
            <li><strong>Data quality:</strong> Remove rows with negative or zero quantities/prices</li>
            <li><strong>File size:</strong> Try with sample data first to test the app</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
