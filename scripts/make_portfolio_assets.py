#!/usr/bin/env python3
"""
Portfolio Assets Generator
Creates professional figures and assets for project showcase.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

try:
    from app_core import compute_rfm, scale_rfm, run_kmeans, evaluate_clusters, pca_2d
    # Create a wrapper function for k selection
    def run_kmeans_with_selection(rfm_scaled, k_range=(2, 6)):
        """KMeans with k selection using app_core functions."""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        results = {}
        best_k = 2
        best_silhouette = -1
        
        for k in range(k_range[0], k_range[1] + 1):
            labels, kmeans = run_kmeans(rfm_scaled, k)
            
            features = rfm_scaled[["Recency", "Frequency", "Monetary"]].values
            silhouette = silhouette_score(features, labels)
            calinski = calinski_harabasz_score(features, labels)
            davies = davies_bouldin_score(features, labels)
            
            results[k] = {
                'labels': labels,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies,
                'inertia': kmeans.inertia_
            }
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
        
        return results, best_k
except ImportError:
    print("Warning: Could not import from src.app_core. Using fallback functions.")
    
    def compute_rfm(df):
        """Fallback RFM computation."""
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
        
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (datetime.now() - x.max()).days,
            'InvoiceNo': 'count',
            'TotalAmount': 'sum'
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        return rfm
    
    def scale_rfm(rfm_df):
        """Fallback RFM scaling."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        rfm_scaled = pd.DataFrame(
            scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']]),
            columns=['Recency', 'Frequency', 'Monetary'],
            index=rfm_df.index
        )
        return rfm_scaled
    
    def run_kmeans_with_selection(features, k_range=(2, 6)):
        """Fallback KMeans with k selection."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        results = {}
        best_k = 2
        best_silhouette = -1
        
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
            labels = kmeans.fit_predict(features)
            
            silhouette = silhouette_score(features, labels)
            calinski = calinski_harabasz_score(features, labels)
            davies = davies_bouldin_score(features, labels)
            
            results[k] = {
                'labels': labels,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies,
                'inertia': kmeans.inertia_
            }
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
        
        return results, best_k
    
    def pca_2d(features):
        """Fallback PCA to 2D."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(features)
        return pca_features, pca

def create_hero_pca_scatter(rfm_scaled, labels, pca_features, output_path):
    """Create PCA scatter plot with centroids."""
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                         c=labels, cmap='viridis', alpha=0.7, s=50)
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        centroid = pca_features[mask].mean(axis=0)
        plt.scatter(centroid[0], centroid[1], 
                   c=[plt.cm.viridis(label / max(unique_labels))], 
                   s=200, marker='x', linewidths=3, label=f'Segment {label}')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Customer Segments - PCA Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_hero_rfm_heatmap(rfm_df, labels, output_path):
    """Create RFM heatmap by segment."""
    rfm_with_labels = rfm_df.copy()
    rfm_with_labels['segment'] = labels
    
    segment_means = rfm_with_labels.groupby('segment')[['Recency', 'Frequency', 'Monetary']].mean()
    segment_means_norm = (segment_means - segment_means.min()) / (segment_means.max() - segment_means.min())
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(segment_means_norm.T, cmap='RdYlBu_r', aspect='auto')
    
    for i in range(len(segment_means_norm)):
        for j in range(len(segment_means_norm.columns)):
            text = plt.text(i, j, f'{segment_means.iloc[i, j]:.1f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, label='Normalized Value')
    plt.xlabel('Segment')
    plt.ylabel('RFM Metric')
    plt.title('Average RFM Values by Segment')
    plt.xticks(range(len(segment_means)), [f'Segment {i}' for i in segment_means.index])
    plt.yticks(range(len(segment_means.columns)), segment_means.columns)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_hero_segments_size(labels, output_path):
    """Create segment size bar chart."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(unique)), counts, color='skyblue', alpha=0.7)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        percentage = (count / total) * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*total,
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.title('Customer Distribution by Segment')
    plt.xticks(range(len(unique)), [f'Segment {i}' for i in unique])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_hero_revenue_share(rfm_df, labels, output_path):
    """Create revenue share bar chart."""
    rfm_with_labels = rfm_df.copy()
    rfm_with_labels['segment'] = labels
    
    segment_revenue = rfm_with_labels.groupby('segment')['Monetary'].sum()
    total_revenue = segment_revenue.sum()
    revenue_share = (segment_revenue / total_revenue) * 100
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(revenue_share)), revenue_share.values, 
                   color='lightgreen', alpha=0.7)
    
    for i, (bar, share) in enumerate(zip(bars, revenue_share.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{share:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Segment')
    plt.ylabel('Revenue Share (%)')
    plt.title('Revenue Distribution by Segment')
    plt.xticks(range(len(revenue_share)), [f'Segment {i}' for i in revenue_share.index])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_quicklook_gif(figure_paths, output_path):
    """Create animated GIF from available figures."""
    try:
        import imageio
    except ImportError:
        print("Warning: imageio not available. Skipping GIF creation.")
        return False
    
    existing_files = [f for f in figure_paths if os.path.exists(f)]
    
    if len(existing_files) < 2:
        print("Warning: Not enough figure files for GIF. Skipping.")
        return False
    
    try:
        images = []
        for file_path in existing_files:
            img = imageio.imread(file_path)
            images.append(img)
        
        imageio.mimsave(output_path, images, duration=2.0)
        print(f"GIF created: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return False

def main():
    """Main execution function."""
    print("Generating portfolio assets...")
    
    os.makedirs('reports/figures', exist_ok=True)
    
    data_path = 'data/sample/online_retail_sample.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run scripts/generate_sample_data.py first.")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} transactions with {df['CustomerID'].nunique()} customers")
    
    print("Computing RFM metrics...")
    rfm_df = compute_rfm(df)
    
    print("Scaling RFM features...")
    rfm_scaled, scaler = scale_rfm(rfm_df)
    
    print("Running KMeans clustering...")
    results, best_k = run_kmeans_with_selection(rfm_scaled, k_range=(2, 6))
    
    print(f"Selected k={best_k} clusters")
    
    final_labels = results[best_k]['labels']
    
    print("Creating PCA visualization...")
    pca_features, pca = pca_2d(rfm_scaled)
    
    print("Generating portfolio figures...")
    
    create_hero_pca_scatter(rfm_scaled, final_labels, pca_features, 
                           'reports/figures/hero_pca_scatter.png')
    
    create_hero_rfm_heatmap(rfm_df, final_labels, 
                           'reports/figures/hero_rfm_heatmap.png')
    
    create_hero_segments_size(final_labels, 
                             'reports/figures/hero_segments_size.png')
    
    create_hero_revenue_share(rfm_df, final_labels, 
                             'reports/figures/hero_revenue_share.png')
    
    print("Creating segment profiles...")
    rfm_with_labels = rfm_df.copy()
    rfm_with_labels['segment'] = final_labels
    
    segment_profiles = rfm_with_labels.groupby('segment').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    
    segment_profiles.columns = ['segment', 'n_customers', 'avg_recency', 'avg_frequency', 'avg_monetary']
    
    total_customers = segment_profiles['n_customers'].sum()
    total_revenue = rfm_with_labels['Monetary'].sum()
    
    segment_profiles['share_customers'] = segment_profiles['n_customers'] / total_customers
    segment_profiles['revenue_share'] = (segment_profiles['avg_monetary'] * segment_profiles['n_customers']) / total_revenue
    
    assert abs(segment_profiles['share_customers'].sum() - 1.0) < 1e-6, "Customer shares don't sum to 1"
    assert abs(segment_profiles['revenue_share'].sum() - 1.0) < 1e-6, "Revenue shares don't sum to 1"
    
    segment_profiles.to_csv('reports/figures/portfolio_segment_profiles.csv', index=False)
    
    print("Creating animated GIF...")
    figure_paths = [
        'reports/figures/hero_pca_scatter.png',
        'reports/figures/hero_rfm_heatmap.png',
        'reports/figures/hero_segments_size.png',
        'reports/figures/hero_revenue_share.png'
    ]
    
    gif_created = create_quicklook_gif(figure_paths, 'reports/figures/quicklook.gif')
    
    print("\nPortfolio assets generated:")
    print("- hero_pca_scatter.png")
    print("- hero_rfm_heatmap.png")
    print("- hero_segments_size.png")
    print("- hero_revenue_share.png")
    print("- portfolio_segment_profiles.csv")
    if gif_created:
        print("- quicklook.gif")
    else:
        print("- quicklook.gif (skipped - imageio or sources missing)")
    
    print(f"\nBest clustering: k={best_k}")
    print(f"Silhouette score: {results[best_k]['silhouette']:.3f}")

if __name__ == "__main__":
    main()
