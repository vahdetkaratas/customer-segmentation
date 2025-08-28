#!/usr/bin/env python3
"""
Alternative Clustering Methods: DBSCAN and Hierarchical Clustering.
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
    from app_core import compute_rfm, scale_rfm, run_dbscan, run_hierarchical, evaluate_clusters, pca_2d
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
    
    def run_dbscan(rfm_scaled, eps, min_samples):
        """Fallback DBSCAN clustering."""
        from sklearn.cluster import DBSCAN
        features = rfm_scaled[["Recency", "Frequency", "Monetary"]].values
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        return labels, dbscan
    
    def run_hierarchical(rfm_scaled, n_clusters):
        """Fallback Hierarchical clustering."""
        from sklearn.cluster import AgglomerativeClustering
        features = rfm_scaled[["Recency", "Frequency", "Monetary"]].values
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = hierarchical.fit_predict(features)
        return labels, hierarchical
    
    def pca_2d(features):
        """Fallback PCA to 2D."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(features)
        return pca_features, pca

def run_dbscan_analysis(rfm_scaled, output_dir='reports/figures'):
    """Run DBSCAN analysis with parameter grid search."""
    print("Running DBSCAN analysis...")
    
    # Parameter grid
    eps_values = [0.3, 0.5, 0.7, 1.0]
    min_samples_values = [3, 5, 10]
    
    results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                labels, dbscan = run_dbscan(rfm_scaled, eps, min_samples)
                
                # Count clusters (excluding noise)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                # Calculate silhouette score (excluding noise points)
                from sklearn.metrics import silhouette_score
                if n_clusters > 1 and -1 in labels:
                    non_noise_mask = labels != -1
                    if sum(non_noise_mask) > 1:
                        silhouette = silhouette_score(
                            rfm_scaled[["Recency", "Frequency", "Monetary"]].values[non_noise_mask],
                            labels[non_noise_mask]
                        )
                    else:
                        silhouette = np.nan
                elif n_clusters > 1:
                    silhouette = silhouette_score(
                        rfm_scaled[["Recency", "Frequency", "Monetary"]].values,
                        labels
                    )
                else:
                    silhouette = np.nan
                
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'silhouette': silhouette,
                    'labels': labels
                })
                
                print(f"DBSCAN (eps={eps}, min_samples={min_samples}): {n_clusters} clusters, silhouette={silhouette:.3f}")
                
            except Exception as e:
                print(f"Error with DBSCAN (eps={eps}, min_samples={min_samples}): {e}")
    
    # Find best configuration
    valid_results = [r for r in results if not np.isnan(r['silhouette'])]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['silhouette'])
        print(f"\nBest DBSCAN: eps={best_result['eps']}, min_samples={best_result['min_samples']}")
        print(f"Clusters: {best_result['n_clusters']}, Silhouette: {best_result['silhouette']:.3f}")
        
        # Save best labels
        labels_df = pd.DataFrame({
            'CustomerID': rfm_scaled.index,
            'segment': best_result['labels']
        })
        labels_df.to_csv('data/processed/dbscan_labels.csv', index=False)
        
        # Save model
        import joblib
        _, best_dbscan = run_dbscan(rfm_scaled, best_result['eps'], best_result['min_samples'])
        joblib.dump(best_dbscan, 'models/dbscan_best.joblib')
        
        # Create PCA visualization
        pca_features, pca = pca_2d(rfm_scaled[["Recency", "Frequency", "Monetary"]].values)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                            c=best_result['labels'], cmap='viridis', alpha=0.7, s=50)
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'DBSCAN Clustering - PCA Visualization')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/dbscan_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return best_result
    else:
        print("No valid DBSCAN configurations found")
        return None

def run_hierarchical_analysis(rfm_scaled, output_dir='reports/figures'):
    """Run Hierarchical clustering analysis."""
    print("Running Hierarchical clustering analysis...")
    
    # Test different numbers of clusters
    n_clusters_range = range(2, 7)
    results = []
    
    for n_clusters in n_clusters_range:
        try:
            labels, hierarchical = run_hierarchical(rfm_scaled, n_clusters)
            
            # Calculate metrics
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            silhouette = silhouette_score(
                rfm_scaled[["Recency", "Frequency", "Monetary"]].values,
                labels
            )
            calinski = calinski_harabasz_score(
                rfm_scaled[["Recency", "Frequency", "Monetary"]].values,
                labels
            )
            
            results.append({
                'n_clusters': n_clusters,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'labels': labels
            })
            
            print(f"Hierarchical (k={n_clusters}): silhouette={silhouette:.3f}, calinski={calinski:.3f}")
            
        except Exception as e:
            print(f"Error with Hierarchical (k={n_clusters}): {e}")
    
    if results:
        # Find best configuration (highest silhouette)
        best_result = max(results, key=lambda x: x['silhouette'])
        print(f"\nBest Hierarchical: k={best_result['n_clusters']}")
        print(f"Silhouette: {best_result['silhouette']:.3f}, Calinski-Harabasz: {best_result['calinski_harabasz']:.3f}")
        
        # Save best labels
        labels_df = pd.DataFrame({
            'CustomerID': rfm_scaled.index,
            'segment': best_result['labels']
        })
        labels_df.to_csv('data/processed/hierarchical_labels.csv', index=False)
        
        # Save model
        import joblib
        _, best_hierarchical = run_hierarchical(rfm_scaled, best_result['n_clusters'])
        joblib.dump(best_hierarchical, f'models/hierarchical_k{best_result["n_clusters"]}.joblib')
        
        # Create PCA visualization
        pca_features, pca = pca_2d(rfm_scaled[["Recency", "Frequency", "Monetary"]].values)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                            c=best_result['labels'], cmap='viridis', alpha=0.7, s=50)
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'Hierarchical Clustering - PCA Visualization (k={best_result["n_clusters"]})')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/hierarchical_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return best_result
    else:
        print("No valid Hierarchical configurations found")
        return None

def create_comparison_table(kmeans_result, dbscan_result, hierarchical_result, output_dir='reports/figures'):
    """Create comparison table of all clustering methods."""
    print("Creating comparison table...")
    
    comparison_data = []
    
    # KMeans results
    if kmeans_result:
        comparison_data.append({
            'Method': 'KMeans',
            'Clusters': kmeans_result.get('n_clusters', len(set(kmeans_result['labels']))),
            'Silhouette': kmeans_result.get('silhouette', np.nan),
            'Calinski-Harabasz': kmeans_result.get('calinski_harabasz', np.nan),
            'Davies-Bouldin': kmeans_result.get('davies_bouldin', np.nan)
        })
    
    # DBSCAN results
    if dbscan_result:
        comparison_data.append({
            'Method': 'DBSCAN',
            'Clusters': dbscan_result['n_clusters'],
            'Silhouette': dbscan_result['silhouette'],
            'Calinski-Harabasz': np.nan,
            'Davies-Bouldin': np.nan
        })
    
    # Hierarchical results
    if hierarchical_result:
        comparison_data.append({
            'Method': 'Hierarchical',
            'Clusters': hierarchical_result['n_clusters'],
            'Silhouette': hierarchical_result['silhouette'],
            'Calinski-Harabasz': hierarchical_result['calinski_harabasz'],
            'Davies-Bouldin': np.nan
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f'{output_dir}/clustering_comparison.csv', index=False)
    
    print("\nClustering Comparison:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

def main():
    """Main execution function."""
    print("Running Alternative Clustering Analysis...")
    
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
    
    # Load KMeans results for comparison
    kmeans_result = None
    kmeans_labels_path = 'data/processed/rfm_with_segment.csv'
    if os.path.exists(kmeans_labels_path):
        kmeans_df = pd.read_csv(kmeans_labels_path)
        if 'segment' in kmeans_df.columns:
            kmeans_labels = kmeans_df['segment'].values
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            features = rfm_scaled[["Recency", "Frequency", "Monetary"]].values
            kmeans_result = {
                'labels': kmeans_labels,
                'n_clusters': len(set(kmeans_labels)),
                'silhouette': silhouette_score(features, kmeans_labels),
                'calinski_harabasz': calinski_harabasz_score(features, kmeans_labels),
                'davies_bouldin': davies_bouldin_score(features, kmeans_labels)
            }
    
    # Run DBSCAN analysis
    dbscan_result = run_dbscan_analysis(rfm_scaled)
    
    # Run Hierarchical analysis
    hierarchical_result = run_hierarchical_analysis(rfm_scaled)
    
    # Create comparison table
    comparison_df = create_comparison_table(kmeans_result, dbscan_result, hierarchical_result)
    
    # Merge labels with RFM data
    if dbscan_result:
        rfm_path = 'data/processed/rfm_table.csv'
        if os.path.exists(rfm_path):
            rfm_df = pd.read_csv(rfm_path)
            rfm_with_dbscan = rfm_df.copy()
            rfm_with_dbscan['segment'] = dbscan_result['labels']
            rfm_with_dbscan.to_csv('data/processed/rfm_with_dbscan.csv', index=False)
    
    if hierarchical_result:
        rfm_path = 'data/processed/rfm_table.csv'
        if os.path.exists(rfm_path):
            rfm_df = pd.read_csv(rfm_path)
            rfm_with_hierarchical = rfm_df.copy()
            rfm_with_hierarchical['segment'] = hierarchical_result['labels']
            rfm_with_hierarchical.to_csv('data/processed/rfm_with_hierarchical.csv', index=False)
    
    print("\nAlternative clustering analysis complete!")
    print("Files created:")
    if dbscan_result:
        print("- data/processed/dbscan_labels.csv")
        print("- data/processed/rfm_with_dbscan.csv")
        print("- models/dbscan_best.joblib")
        print("- reports/figures/dbscan_scatter.png")
    if hierarchical_result:
        print("- data/processed/hierarchical_labels.csv")
        print("- data/processed/rfm_with_hierarchical.csv")
        print(f"- models/hierarchical_k{hierarchical_result['n_clusters']}.joblib")
        print("- reports/figures/hierarchical_scatter.png")
    print("- reports/figures/clustering_comparison.csv")

if __name__ == "__main__":
    main()
