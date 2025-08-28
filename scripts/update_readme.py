#!/usr/bin/env python3
"""
Update README.md with Step 8: Portfolio Assets & Quick Demo section.
"""

import os
import re

def update_readme():
    """Update README.md with Step 8 section."""
    readme_path = 'README.md'
    
    if not os.path.exists(readme_path):
        print(f"README.md not found at {readme_path}")
        return
    
    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if Step 8 already exists
    if "Step 8: Portfolio Assets" in content:
        print("Step 8 section already exists in README.md")
        return
    
    # Find the end of the Future Work section to insert Step 8
    future_work_pattern = r"(## Future Work.*?)(## Contributing)"
    match = re.search(future_work_pattern, content, re.DOTALL)
    
    if not match:
        print("Could not find Future Work section in README.md")
        return
    
    future_work_section = match.group(1)
    contributing_section = match.group(2)
    
    # Create Step 8 content
    step8_content = """
### Step 8: Portfolio Assets & Quick Demo

This step creates professional portfolio assets and provides a quick demo experience for showcasing the project.

#### Sample Datasets
- **Main Dataset**: `data/sample/online_retail_sample.csv` (~600 rows, 143 customers)
- **Tiny Dataset**: `data/sample/online_retail_tiny.csv` (~20 rows, 8 customers)
- **Schema**: CustomerID, InvoiceDate, InvoiceNo, Quantity, UnitPrice
- **Features**: Deterministic generation with realistic customer behavior patterns

#### Portfolio Figures
- **PCA Scatter**: `reports/figures/hero_pca_scatter.png` - Customer segments visualization
- **RFM Heatmap**: `reports/figures/hero_rfm_heatmap.png` - Average RFM values by segment
- **Segment Sizes**: `reports/figures/hero_segments_size.png` - Customer distribution by segment
- **Revenue Share**: `reports/figures/hero_revenue_share.png` - Revenue distribution by segment

#### Quick Demo GIF
- **Animated Overview**: `reports/figures/quicklook.gif` - Stitched portfolio figures
- **Requirements**: `imageio` package for GIF generation
- **Fallback**: Graceful skip if imageio not available

#### Try Quickly with Streamlit
```bash
# Install dependencies
make install

# Run Streamlit app
streamlit run streamlit_app/app.py

# In the app: check "Use sample data" for instant demo
```

#### Regenerate Assets
```bash
# Generate all portfolio assets
make assets

# Or run directly
python scripts/make_portfolio_assets.py
```

#### Segment Profiles
- **CSV Output**: `reports/figures/portfolio_segment_profiles.csv`
- **Columns**: segment, n_customers, share_customers, revenue_share, avg_recency, avg_frequency, avg_monetary
- **Validation**: Customer and revenue shares sum to 1.0 (within 1e-6 tolerance)

"""
    
    # Insert Step 8 before Contributing section
    new_content = content.replace(
        match.group(0),
        future_work_section + step8_content + contributing_section
    )
    
    # Update Table of Contents if it exists
    toc_pattern = r"(## Table of Contents.*?)(## )"
    toc_match = re.search(toc_pattern, new_content, re.DOTALL)
    
    if toc_match:
        toc_section = toc_match.group(1)
        next_section = toc_match.group(2)
        
        # Add Step 8 to TOC
        if "Step 8" not in toc_section:
            # Find the last step in TOC
            last_step_pattern = r"(\d+\. \[.*?\]\(.*?\)\s*$)"
            last_step_match = re.search(last_step_pattern, toc_section, re.MULTILINE)
            
            if last_step_match:
                step8_toc = "\n8. [Step 8: Portfolio Assets & Quick Demo](#step-8-portfolio-assets--quick-demo)\n"
                new_toc_section = toc_section + step8_toc
                new_content = new_content.replace(toc_section, new_toc_section)
    
    # Write updated README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("README.md updated with Step 8 section")

def add_hero_section():
    """Add hero section at the top of README."""
    readme_path = 'README.md'
    
    if not os.path.exists(readme_path):
        print(f"README.md not found at {readme_path}")
        return
    
    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if hero section already exists
    if "From raw transactions to actionable segments" in content:
        print("Hero section already exists in README.md")
        return
    
    # Find the start of the content after badges
    badges_pattern = r"(# Customer Segmentation Project.*?)(## Project Description)"
    match = re.search(badges_pattern, content, re.DOTALL)
    
    if not match:
        print("Could not find project title in README.md")
        return
    
    badges_section = match.group(1)
    project_description = match.group(2)
    
    # Create hero section
    hero_content = """
# Customer Segmentation — RFM + Clustering

**From raw transactions to actionable segments in minutes.**

![Customer Segments](reports/figures/hero_pca_scatter.png)

**[Try it now with Streamlit →](#step-6-streamlit-demo-try-it-locally)**

---
"""
    
    # Insert hero section
    new_content = content.replace(
        match.group(0),
        badges_section + hero_content + project_description
    )
    
    # Write updated README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("README.md updated with hero section")

def main():
    """Main function to update README."""
    print("Updating README.md...")
    
    # Add hero section
    add_hero_section()
    
    # Add Step 8 section
    update_readme()
    
    print("README.md update complete!")

if __name__ == "__main__":
    main()
