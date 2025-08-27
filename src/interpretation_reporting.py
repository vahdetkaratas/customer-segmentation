#!/usr/bin/env python3
"""
Segment Profiling & Reporting Script
Step 5 of Customer Segmentation Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import os

# Set configuration
CLUSTER_SOURCE = 'kmeans'  # options: 'kmeans', 'dbscan', 'hierarchical'

# Set plotting style
plt.style.use('default')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')

def load_clustering_data(source):
    """Load clustering data based on source"""
    if source == 'kmeans':
        file_path = 'data/processed/rfm_with_segment.csv'
        if not Path(file_path).exists():
            raise FileNotFoundError(f'KMeans clustering file not found: {file_path}. Please run Step 3 (KMeans clustering) first.')
        df = pd.read_csv(file_path)
        print(f'Loaded KMeans clustering data: {len(df)} customers')
        
    elif source == 'dbscan':
        file_path = 'data/processed/rfm_with_dbscan.csv'
        if not Path(file_path).exists():
            raise FileNotFoundError(f'DBSCAN clustering file not found: {file_path}. Please run Step 4 (Alternative clustering) first.')
        df = pd.read_csv(file_path)
        # Rename dbscan_label to segment if it exists
        if 'dbscan_label' in df.columns:
            df = df.rename(columns={'dbscan_label': 'segment'})
        print(f'Loaded DBSCAN clustering data: {len(df)} customers')
        
    elif source == 'hierarchical':
        file_path = 'data/processed/rfm_with_hierarchical.csv'
        if not Path(file_path).exists():
            raise FileNotFoundError(f'Hierarchical clustering file not found: {file_path}. Please run Step 4 (Alternative clustering) first.')
        df = pd.read_csv(file_path)
        # Rename hierarchical_label to segment if it exists
        if 'hierarchical_label' in df.columns:
            df = df.rename(columns={'hierarchical_label': 'segment'})
        print(f'Loaded Hierarchical clustering data: {len(df)} customers')
        
    else:
        raise ValueError(f'Invalid clustering source: {source}. Use "kmeans", "dbscan", or "hierarchical"')
    
    # Ensure segment column exists
    if 'segment' not in df.columns:
        raise ValueError('No segment column found in the loaded data')
    
    return df

def calculate_segment_profiles(df):
    """Calculate segment profiling statistics"""
    # Calculate overall statistics for churn proxy
    overall_recency_75th = df['Recency'].quantile(0.75)
    
    # Group by segment and calculate statistics
    profiles = []
    
    for segment in sorted(df['segment'].unique()):
        segment_data = df[df['segment'] == segment]
        
        profile = {
            'segment': segment,
            'n_customers': len(segment_data),
            'share_customers': len(segment_data) / len(df),
            'avg_recency': segment_data['Recency'].mean(),
            'median_recency': segment_data['Recency'].median(),
            'avg_frequency': segment_data['Frequency'].mean(),
            'median_frequency': segment_data['Frequency'].median(),
            'avg_monetary': segment_data['Monetary'].mean(),
            'median_monetary': segment_data['Monetary'].median(),
            'revenue_share': segment_data['Monetary'].sum() / df['Monetary'].sum(),
            'high_recency_share': (segment_data['Recency'] > overall_recency_75th).mean()
        }
        profiles.append(profile)
    
    return pd.DataFrame(profiles)

def assign_business_labels(df):
    """Assign business labels based on normalized RFM values"""
    # Normalize RFM values (min-max scaling per column)
    rfm_cols = ['Recency', 'Frequency', 'Monetary']
    df_normalized = df.copy()
    
    for col in rfm_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        df_normalized[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
    
    # Apply business labeling rules
    labels = []
    for _, row in df_normalized.iterrows():
        r_norm = row['Recency_norm']
        f_norm = row['Frequency_norm']
        m_norm = row['Monetary_norm']
        
        # Business labeling logic
        if m_norm >= 0.7 and f_norm >= 0.6 and r_norm <= 0.4:
            label = 'VIP Loyal'
        elif m_norm >= 0.7 and f_norm < 0.6 and r_norm <= 0.4:
            label = 'High-Value Rare Buyers'
        elif 0.3 <= m_norm <= 0.7 and 0.3 <= f_norm <= 0.7 and 0.3 <= r_norm <= 0.7:
            label = 'Regulars'
        elif r_norm >= 0.7 and f_norm < 0.6:
            label = 'At-Risk / Inactive'
        else:
            label = 'General'
        
        labels.append(label)
    
    return labels

def generate_segment_narratives(profiles_df, labeled_df):
    """Generate segment narratives with descriptions and campaign ideas"""
    narratives = []
    
    for _, profile in profiles_df.iterrows():
        segment = profile['segment']
        segment_data = labeled_df[labeled_df['segment'] == segment]
        
        # Get most common business label for this segment
        most_common_label = segment_data['segment_label'].mode().iloc[0]
        
        # Template-based description
        if most_common_label == 'VIP Loyal':
            description = f'Segment {segment} represents our most valuable and loyal customers. They make frequent purchases with high monetary value and have recent activity, indicating strong engagement.'
            campaign_ideas = 'Loyalty rewards program, exclusive early access to new products, VIP customer service hotline'
        elif most_common_label == 'High-Value Rare Buyers':
            description = f'Segment {segment} consists of customers who make high-value purchases but buy infrequently. They may be occasional luxury buyers or seasonal customers.'
            campaign_ideas = 'Premium product launches, seasonal promotions, personalized recommendations for high-value items'
        elif most_common_label == 'Regulars':
            description = f'Segment {segment} represents regular customers with moderate RFM values. They form the backbone of our customer base with consistent but moderate engagement.'
            campaign_ideas = 'Regular newsletter, moderate discount promotions, product recommendations based on purchase history'
        elif most_common_label == 'At-Risk / Inactive':
            description = f'Segment {segment} includes customers who haven\'t purchased recently and have low frequency. They are at risk of churning and need re-engagement efforts.'
            campaign_ideas = 'Win-back campaigns, special reactivation offers, feedback surveys to understand reasons for inactivity'
        else:  # General
            description = f'Segment {segment} represents a diverse group of customers with mixed RFM characteristics. They may include new customers or those with irregular purchasing patterns.'
            campaign_ideas = 'Welcome campaigns for new customers, general promotions, cross-selling opportunities'
        
        narrative = {
            'segment': segment,
            'segment_label': most_common_label,
            'short_description': description,
            'campaign_ideas': campaign_ideas
        }
        narratives.append(narrative)
    
    return pd.DataFrame(narratives)

def create_visualizations(segment_profiles, data, cluster_source):
    """Create all visualizations for stakeholders"""
    # Ensure reports/figures directory exists
    Path('reports/figures').mkdir(parents=True, exist_ok=True)
    
    # 1. Segment size bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(segment_profiles['segment'], segment_profiles['n_customers'], 
                   color=plt.cm.Set3(np.linspace(0, 1, len(segment_profiles))))
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.title(f'Segment Sizes ({cluster_source.title()} Clustering)')
    
    # Add share annotations
    for i, (bar, share) in enumerate(zip(bars, segment_profiles['share_customers'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{share:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('reports/figures/segments_size_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Segment size bar chart saved')
    
    # 2. Revenue share bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(segment_profiles['segment'], segment_profiles['revenue_share'], 
                   color=plt.cm.Set3(np.linspace(0, 1, len(segment_profiles))))
    plt.xlabel('Segment')
    plt.ylabel('Revenue Share')
    plt.title(f'Segment Revenue Share ({cluster_source.title()} Clustering)')
    
    # Add percentage annotations
    for i, (bar, share) in enumerate(zip(bars, segment_profiles['revenue_share'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                 f'{share:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('reports/figures/segments_revenue_share.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Revenue share bar chart saved')
    
    # 3. RFM heatmap
    plt.figure(figsize=(8, 6))
    heatmap_data = segment_profiles[['avg_recency', 'avg_frequency', 'avg_monetary']].T
    heatmap_data.columns = [f'Segment {seg}' for seg in segment_profiles['segment']]
    
    # Normalize for heatmap (min-max scaling)
    heatmap_data_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    
    sns.heatmap(heatmap_data_norm, annot=heatmap_data.round(2), fmt='.2f', 
                cmap='RdYlBu_r', cbar_kws={'label': 'Normalized Value'})
    plt.title(f'Segment RFM Profile Heatmap ({cluster_source.title()} Clustering)')
    plt.xlabel('Segment')
    plt.ylabel('RFM Metric')
    plt.tight_layout()
    plt.savefig('reports/figures/segments_rfm_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ RFM heatmap saved')
    
    # 4. RFM boxplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Recency boxplot
    sns.boxplot(data=data, x='segment', y='Recency', ax=axes[0])
    axes[0].set_title('Recency by Segment')
    axes[0].set_xlabel('Segment')
    axes[0].set_ylabel('Recency (days)')
    
    # Frequency boxplot
    sns.boxplot(data=data, x='segment', y='Frequency', ax=axes[1])
    axes[1].set_title('Frequency by Segment')
    axes[1].set_xlabel('Segment')
    axes[1].set_ylabel('Frequency (purchases)')
    
    # Monetary boxplot
    sns.boxplot(data=data, x='segment', y='Monetary', ax=axes[2])
    axes[2].set_title('Monetary by Segment')
    axes[2].set_xlabel('Segment')
    axes[2].set_ylabel('Monetary ($)')
    
    plt.suptitle(f'RFM Distributions by Segment ({cluster_source.title()} Clustering)', fontsize=16)
    plt.tight_layout()
    plt.savefig('reports/figures/segments_rfm_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ RFM boxplots saved')

def generate_markdown_report(cluster_source, profiles_df, narratives_df):
    """Generate comprehensive markdown report"""
    report_lines = []
    report_lines.append('# Customer Segmentation Report')
    report_lines.append('')
    report_lines.append(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report_lines.append(f'**Clustering Method:** {cluster_source.title()}')
    report_lines.append(f'**Number of Segments:** {len(profiles_df)}')
    report_lines.append('')
    report_lines.append('## Executive Summary')
    report_lines.append('')
    report_lines.append(f'This report presents the results of customer segmentation analysis using {cluster_source.title()} clustering.')
    report_lines.append(f'The analysis identified {len(profiles_df)} distinct customer segments based on RFM (Recency, Frequency, Monetary) metrics.')
    report_lines.append('')
    report_lines.append('## Segment Overview')
    report_lines.append('')
    report_lines.append('### Segment Profiles')
    report_lines.append('')
    report_lines.append('| Segment | Customers | Share | Avg Recency | Avg Frequency | Avg Monetary | Revenue Share | At-Risk % |')
    report_lines.append('|---------|-----------|-------|-------------|---------------|--------------|--------------|-----------|')
    
    for _, profile in profiles_df.iterrows():
        line = f"| {profile['segment']} | {profile['n_customers']:,} | {profile['share_customers']:.1%} | {profile['avg_recency']:.0f} days | {profile['avg_frequency']:.1f} | ${profile['avg_monetary']:,.0f} | {profile['revenue_share']:.1%} | {profile['high_recency_share']:.1%} |"
        report_lines.append(line)
    
    report_lines.append('')
    report_lines.append('## Visualizations')
    report_lines.append('')
    report_lines.append('### Segment Sizes')
    report_lines.append('![Segment Sizes](figures/segments_size_bar.png)')
    report_lines.append('')
    report_lines.append('### Revenue Share by Segment')
    report_lines.append('![Revenue Share](figures/segments_revenue_share.png)')
    report_lines.append('')
    report_lines.append('### RFM Profile Heatmap')
    report_lines.append('![RFM Heatmap](figures/segments_rfm_heatmap.png)')
    report_lines.append('')
    report_lines.append('### RFM Distributions')
    report_lines.append('![RFM Boxplots](figures/segments_rfm_boxplots.png)')
    report_lines.append('')
    report_lines.append('## Key Insights')
    report_lines.append('')
    report_lines.append('### Interpretation Checklist')
    report_lines.append('')
    report_lines.append('**What stands out:**')
    
    # Add insights based on data
    max_revenue_segment = profiles_df.loc[profiles_df['revenue_share'].idxmax()]
    max_customers_segment = profiles_df.loc[profiles_df['n_customers'].idxmax()]
    high_risk_segment = profiles_df.loc[profiles_df['high_recency_share'].idxmax()]
    
    report_lines.append('')
    report_lines.append(f'- **Highest Revenue Segment:** Segment {max_revenue_segment["segment"]} contributes {max_revenue_segment["revenue_share"]:.1%} of total revenue')
    report_lines.append(f'- **Largest Segment:** Segment {max_customers_segment["segment"]} has {max_customers_segment["n_customers"]:,} customers ({max_customers_segment["share_customers"]:.1%} of total)')
    report_lines.append(f'- **Highest Risk Segment:** Segment {high_risk_segment["segment"]} has {high_risk_segment["high_recency_share"]:.1%} customers at risk of churning')
    report_lines.append('')
    report_lines.append('**VIP Customers:** Look for segments with high monetary value and low recency')
    report_lines.append('**At-Risk Customers:** Focus on segments with high recency (inactive) and low frequency')
    report_lines.append('**Growth Opportunities:** Target segments with medium RFM values for upselling')
    report_lines.append('')
    report_lines.append('## Segment Narratives')
    report_lines.append('')
    
    for _, narrative in narratives_df.iterrows():
        report_lines.append(f'### Segment {narrative["segment"]}: {narrative["segment_label"]}')
        report_lines.append('')
        report_lines.append(f'**Description:** {narrative["short_description"]}')
        report_lines.append('')
        report_lines.append(f'**Campaign Ideas:** {narrative["campaign_ideas"]}')
        report_lines.append('')
        report_lines.append('---')
        report_lines.append('')
    
    report_lines.append('## Recommendations')
    report_lines.append('')
    report_lines.append('1. **Immediate Actions:**')
    report_lines.append('   - Implement targeted campaigns for at-risk segments')
    report_lines.append('   - Develop loyalty programs for high-value customers')
    report_lines.append('   - Create re-engagement strategies for inactive customers')
    report_lines.append('')
    report_lines.append('2. **Strategic Initiatives:**')
    report_lines.append('   - Monitor segment evolution over time')
    report_lines.append('   - A/B test different approaches for each segment')
    report_lines.append('   - Develop segment-specific product offerings')
    report_lines.append('')
    report_lines.append('3. **Next Steps:**')
    report_lines.append('   - Set up automated segment monitoring')
    report_lines.append('   - Implement real-time customer scoring')
    report_lines.append('   - Develop predictive models for customer lifetime value')
    report_lines.append('')
    report_lines.append('---')
    report_lines.append('*Report generated automatically by Customer Segmentation Analysis Pipeline*')
    
    return '\n'.join(report_lines)

def main():
    """Main execution function"""
    print('Starting Segment Profiling & Reporting Analysis...')
    print(f'Clustering source: {CLUSTER_SOURCE}')
    
    # Part A: Load clustering data
    try:
        data = load_clustering_data(CLUSTER_SOURCE)
        print(f'Data loaded successfully! Shape: {data.shape}')
        print(f'Segments found: {sorted(data["segment"].unique())}')
    except Exception as e:
        print(f'Error loading data: {e}')
        return
    
    # Part B: Segment profiling
    print('\nCalculating segment profiles...')
    segment_profiles = calculate_segment_profiles(data)
    print(f'Segment profiles calculated for {len(segment_profiles)} segments')
    
    # Save segment profiles
    segment_profiles.to_csv('data/processed/segment_profiles.csv', index=False)
    print('✓ Segment profiles saved to: data/processed/segment_profiles.csv')
    
    # Create customer segments final file
    customer_segments_final = data[['CustomerID', 'segment', 'Recency', 'Frequency', 'Monetary']].copy()
    customer_segments_final.to_csv('data/processed/customer_segments_final.csv', index=False)
    print('✓ Customer segments final saved to: data/processed/customer_segments_final.csv')
    
    # Part C: Visualizations
    print('\nCreating visualizations...')
    create_visualizations(segment_profiles, data, CLUSTER_SOURCE)
    
    # Part D: Business labels
    print('\nAssigning business labels...')
    business_labels = assign_business_labels(data)
    data['segment_label'] = business_labels
    
    # Save labeled data
    data_labeled = data[['CustomerID', 'segment', 'segment_label', 'Recency', 'Frequency', 'Monetary']].copy()
    data_labeled.to_csv('data/processed/customer_segments_labeled.csv', index=False)
    print('✓ Customer segments labeled saved to: data/processed/customer_segments_labeled.csv')
    
    # Part E: Segment narratives
    print('\nGenerating segment narratives...')
    segment_narratives = generate_segment_narratives(segment_profiles, data_labeled)
    # Ensure segment column is integer
    segment_narratives['segment'] = segment_narratives['segment'].astype(int)
    segment_narratives.to_csv('data/processed/segment_narratives.csv', index=False)
    print('✓ Segment narratives saved to: data/processed/segment_narratives.csv')
    
    # Part F: Auto-generated report
    print('\nGenerating auto-report...')
    
    # Ensure reports directory exists
    Path('reports').mkdir(exist_ok=True)
    
    # Generate markdown report
    markdown_report = generate_markdown_report(CLUSTER_SOURCE, segment_profiles, segment_narratives)
    
    # Save markdown report
    with open('reports/segment_report.md', 'w') as f:
        f.write(markdown_report)
    print('✓ Markdown report saved to: reports/segment_report.md')
    
    # Generate simple HTML report
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Customer Segmentation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Customer Segmentation Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><strong>Clustering Method:</strong> {CLUSTER_SOURCE.title()}</p>
    <p><strong>Number of Segments:</strong> {len(segment_profiles)}</p>
    
    <h2>Segment Overview</h2>
    {segment_profiles.to_html(index=False)}
    
    <h2>Visualizations</h2>
    <h3>Segment Sizes</h3>
    <img src="figures/segments_size_bar.png" alt="Segment Sizes">
    
    <h3>Revenue Share by Segment</h3>
    <img src="figures/segments_revenue_share.png" alt="Revenue Share">
    
    <h3>RFM Profile Heatmap</h3>
    <img src="figures/segments_rfm_heatmap.png" alt="RFM Heatmap">
    
    <h3>RFM Distributions</h3>
    <img src="figures/segments_rfm_boxplots.png" alt="RFM Boxplots">
    
    <h2>Segment Narratives</h2>
    {segment_narratives.to_html(index=False)}
</body>
</html>'''
    
    with open('reports/segment_report.html', 'w') as f:
        f.write(html_content)
    print('✓ HTML report saved to: reports/segment_report.html')
    
    print('\n🎉 Segment Profiling & Reporting Analysis Completed Successfully!')
    print('\nGenerated Files:')
    print('- data/processed/segment_profiles.csv')
    print('- data/processed/customer_segments_final.csv')
    print('- data/processed/customer_segments_labeled.csv')
    print('- data/processed/segment_narratives.csv')
    print('- reports/figures/segments_size_bar.png')
    print('- reports/figures/segments_revenue_share.png')
    print('- reports/figures/segments_rfm_heatmap.png')
    print('- reports/figures/segments_rfm_boxplots.png')
    print('- reports/segment_report.md')
    print('- reports/segment_report.html')

if __name__ == "__main__":
    main()
