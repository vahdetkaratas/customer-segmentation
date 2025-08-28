#!/usr/bin/env python3
"""
KMeans Clustering with k-selection and validation metrics.
Replaces the notebook functionality with a clean script.
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
    print("Warning: Could not import from app_core. Using fallback functions.")
    
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
        return rfm_scaled, scaler
    
    def run_kmeans_with_selection(rfm_scaled, k_range=(2, 6)):
        """Fallback KMeans with k selection."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        results = {}
        best_k = 2
        best_silhouette = -1
        
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
            labels = kmeans.fit_predict(rfm_scaled[["Recency", "Frequency", "Monetary"]].values)
            
            silhouette = silhouette_score(rfm_scaled[["Recency", "Frequency", "Monetary"]].values, labels)
            calinski = calinski_harabasz_score(rfm_scaled[["Recency", "Frequency", "Monetary"]].values, labels)
            davies = davies_bouldin_score(rfm_scaled[["Recency", "Frequency", "Monetary"]].values, labels)
            
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

def create_validation_plots(results, output_dir='reports/figures'):
    """Create validation plots for k selection."""
    os.makedirs(output_dir, exist_ok=True)
    
    k_values = list(results.keys())
    inertias = [results[k]['inertia'] for k in k_values]
    silhouettes = [results[k]['silhouette'] for k in k_values]
    calinski = [results[k]['calinski_harabasz'] for k in k_values]
    davies = [results[k]['davies_bouldin'] for k in k_values]
    
    # Elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k Selection')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/elbow_inertia.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouettes, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different k Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/silhouette_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined validation metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(k_values, calinski, 'go-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Calinski-Harabasz Index')
    ax1.set_title('Calinski-Harabasz Index')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(k_values, davies, 'mo-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Davies-Bouldin Index')
    ax2.set_title('Davies-Bouldin Index (Lower is Better)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_validity_indices.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main execution function."""
    print("Running KMeans Clustering Analysis...")
    
    # Ensure directories exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    # Load scaled RFM data
    rfm_scaled_path = 'data/processed/rfm_scaled.csv'
    if not os.path.exists(rfm_scaled_path):
        print(f"Error: {rfm_scaled_path} not found. Run Step 2 first.")
        return
    
    rfm_scaled = pd.read_csv(rfm_scaled_path)
    print(f"Loaded scaled RFM data with {len(rfm_scaled)} customers")
    
    # Run KMeans with k selection
    print("Running KMeans clustering with k selection...")
    results, best_k = run_kmeans_with_selection(rfm_scaled, k_range=(2, 10))
    
    print(f"Selected k={best_k} clusters")
    print(f"Silhouette score: {results[best_k]['silhouette']:.3f}")
    print(f"Calinski-Harabasz: {results[best_k]['calinski_harabasz']:.3f}")
    print(f"Davies-Bouldin: {results[best_k]['davies_bouldin']:.3f}")
    
    # Create validation plots
    print("Creating validation plots...")
    create_validation_plots(results)
    
    # Save final labels
    final_labels = results[best_k]['labels']
    labels_df = pd.DataFrame({
        'CustomerID': rfm_scaled.index,
        'segment': final_labels
    })
    labels_df.to_csv(f'data/processed/labels_k{best_k}.csv', index=False)
    
    # Merge with original RFM data
    rfm_path = 'data/processed/rfm_table.csv'
    if os.path.exists(rfm_path):
        rfm_df = pd.read_csv(rfm_path)
        rfm_with_segments = rfm_df.copy()
        rfm_with_segments['segment'] = final_labels
        rfm_with_segments.to_csv(f'data/processed/rfm_with_segment.csv', index=False)
    
    # Save the model
    from sklearn.cluster import KMeans
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20, max_iter=500)
    final_kmeans.fit(rfm_scaled[["Recency", "Frequency", "Monetary"]].values)
    
    import joblib
    joblib.dump(final_kmeans, f'models/kmeans_k{best_k}.joblib')
    
    # Create PCA visualization
    print("Creating PCA visualization...")
    pca_features, pca = pca_2d(rfm_scaled[["Recency", "Frequency", "Monetary"]].values)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                         c=final_labels, cmap='viridis', alpha=0.7, s=50)
    
    unique_labels = np.unique(final_labels)
    for label in unique_labels:
        mask = final_labels == label
        centroid = pca_features[mask].mean(axis=0)
        plt.scatter(centroid[0], centroid[1], 
                   c=[plt.cm.viridis(label / max(unique_labels))], 
                   s=200, marker='x', linewidths=3, label=f'Segment {label}')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Customer Segments - PCA Visualization (k={best_k})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'reports/figures/pca_scatter_k{best_k}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create RFM heatmap
    if os.path.exists(rfm_path):
        rfm_df = pd.read_csv(rfm_path)
        rfm_with_labels = rfm_df.copy()
        rfm_with_labels['segment'] = final_labels
        
        segment_means = rfm_with_labels.groupby('segment')[['Recency', 'Frequency', 'Monetary']].mean()
        segment_means_norm = (segment_means - segment_means.min()) / (segment_means.max() - segment_means.min())
        
        plt.figure(figsize=(8, 6))
        im = plt.imshow(segment_means_norm.T, cmap='RdYlBu_r', aspect='auto')
        
        for i in range(len(segment_means_norm)):
            for j in range(len(segment_means_norm.columns)):
                plt.text(i, j, f'{segment_means.iloc[i, j]:.1f}',
                        ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, label='Normalized Value')
        plt.xlabel('Segment')
        plt.ylabel('RFM Metric')
        plt.title(f'Average RFM Values by Segment (k={best_k})')
        plt.xticks(range(len(segment_means)), [f'Segment {i}' for i in segment_means.index])
        plt.yticks(range(len(segment_means.columns)), segment_means.columns)
        plt.tight_layout()
        plt.savefig(f'reports/figures/heatmap_rfm_by_segment_k{best_k}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\nClustering analysis complete!")
    print(f"Files created:")
    print(f"- data/processed/labels_k{best_k}.csv")
    print(f"- data/processed/rfm_with_segment.csv")
    print(f"- models/kmeans_k{best_k}.joblib")
    print(f"- reports/figures/elbow_inertia.png")
    print(f"- reports/figures/silhouette_scores.png")
    print(f"- reports/figures/cluster_validity_indices.png")
    print(f"- reports/figures/pca_scatter_k{best_k}.png")
    print(f"- reports/figures/heatmap_rfm_by_segment_k{best_k}.png")

if __name__ == "__main__":
    main()
