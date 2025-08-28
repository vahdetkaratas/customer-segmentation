#!/usr/bin/env python3
"""
Clean README.md by removing all notebook references.
"""

import os
import re

def clean_readme():
    """Remove all notebook references from README."""
    readme_path = 'README.md'
    
    if not os.path.exists(readme_path):
        print(f"README.md not found at {readme_path}")
        return
    
    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove notebook references
    patterns_to_remove = [
        r'jupyter notebook notebooks/.*?\.ipynb',
        r'cd notebooks',
        r'jupyter notebook',
        r'notebooks/.*?\.ipynb',
        r'Run Clustering \(via notebook\)',
        r'Run Alternative Clustering \(via notebook\)',
        r'Option 2: Jupyter Notebook',
        r'#### Option 2: Jupyter Notebook.*?```',
    ]
    
    cleaned_content = content
    for pattern in patterns_to_remove:
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up multiple empty lines
    cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
    
    # Update step numbers in the workflow
    # Find and replace step references
    step_mapping = {
        'Step 4: Run Clustering (via notebook)': 'Step 4: Run Clustering',
        'Step 5: Run Alternative Clustering (via notebook)': 'Step 5: Run Alternative Clustering',
    }
    
    for old_step, new_step in step_mapping.items():
        cleaned_content = cleaned_content.replace(old_step, new_step)
    
    # Update the workflow section
    workflow_pattern = r'(#### Option 1: Python Scripts \(Step by Step\).*?)(### Expected Output)'
    match = re.search(workflow_pattern, cleaned_content, re.DOTALL)
    
    if match:
        workflow_section = match.group(1)
        expected_output = match.group(2)
        
        # Clean up the workflow section
        new_workflow = """#### Option 1: Python Scripts (Step by Step)
```bash
# Step 1: RFM Analysis
python src/rfm_analysis.py

# Step 2: Data Scaling
python src/rfm_scaling.py

# Step 3: Test Scaling
pytest -q tests/test_scaling.py

# Step 4: Run Clustering
python src/clustering.py

# Step 5: Run Alternative Clustering
python src/alternative_clustering.py

# Step 6: Run Segment Profiling & Reporting
python src/interpretation_reporting.py

# Step 7: Run All Tests
pytest -q

# Step 8: Generate Portfolio Assets
python scripts/make_portfolio_assets.py

# Step 9: Try the Interactive Streamlit App
streamlit run streamlit_app/app.py
```
"""
        
        cleaned_content = cleaned_content.replace(
            match.group(0),
            new_workflow + expected_output
        )
    
    # Write updated README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print("README.md cleaned - all notebook references removed")

def main():
    """Main function to clean README."""
    print("Cleaning README.md...")
    clean_readme()
    print("README.md cleaning complete!")

if __name__ == "__main__":
    main()
