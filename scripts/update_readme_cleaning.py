#!/usr/bin/env python3
"""
Update README.md to add data cleaning section.
"""

import os
import re

def add_cleaning_section():
    """Add data cleaning section to README."""
    readme_path = 'README.md'
    
    if not os.path.exists(readme_path):
        print(f"README.md not found at {readme_path}")
        return
    
    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if cleaning section already exists
    if "## Step X: Data Cleaning & Validation" in content:
        print("Data cleaning section already exists in README")
        return
    
    # Find the end of the workflow section to insert before
    workflow_pattern = r'(#### Option 1: Python Scripts \(Step by Step\).*?)(### Expected Output)'
    match = re.search(workflow_pattern, content, re.DOTALL)
    
    if match:
        workflow_section = match.group(1)
        expected_output = match.group(2)
        
        # Create new workflow with cleaning step
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

# Step 9: Data Cleaning & Validation (NEW!)
python scripts/run_cleaning.py --input data/sample/online_retail_sample.csv

# Step 10: Try the Interactive Streamlit App
streamlit run streamlit_app/app.py
```
"""
        
        # Insert cleaning section before Expected Output
        cleaning_section = """

## Step X: Data Cleaning & Validation (Analyze first, Apply on approval)

Robust data cleaning with two-phase workflow: **ANALYZE** (detect issues, propose fixes) and **APPLY** (execute approved plan).

### Analyze Mode (Dry-Run)
Detects schema/type issues, duplicates, invalid values, outliers, thin customers. Writes human-readable reports:

- `reports/cleaning_report.md` - Detailed analysis and proposed plan
- `reports/cleaning_findings.json` - Machine-readable findings

### Apply Mode
Executes the approved plan and writes cleaned outputs:

- `data/processed/transactions_cleaned.csv` - Cleaned transaction data
- `data/processed/transactions_rejected.csv` - Rows that were removed
- `data/processed/transactions_issues_catalog.csv` - Detailed issue tracking

### Usage

```bash
# Analyze (no changes)
python scripts/run_cleaning.py --input data/sample/online_retail_sample.csv

# Apply after approval
python scripts/run_cleaning.py --input data/sample/online_retail_sample.csv --apply

# Apply an approved plan exactly
python scripts/run_cleaning.py --input data/sample/online_retail_sample.csv --apply --plan-from path/to/plan.json
```

### Configuration
See `config/cleaning_rules.yml` for thresholds and choices (drop vs winsorize vs flag).

### Tests
```bash
pytest -q tests/test_cleaning.py
```

### Features
- **Schema Validation**: Ensures required columns and data types
- **Type Coercion**: Smart parsing of dates and numeric values
- **Duplicate Detection**: Exact and near-duplicate identification
- **Outlier Handling**: IQR-based detection with winsorization/dropping options
- **Business Rules**: Invalid quantities/prices, thin customers, date range validation
- **Audit Trail**: Complete tracking of what was changed and why
- **Configurable**: YAML-based rules for easy customization

"""
        
        # Replace the workflow section
        new_content = content.replace(
            match.group(0),
            new_workflow + cleaning_section + expected_output
        )
        
        # Write updated README
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("README.md updated with data cleaning section")
    else:
        print("Could not find workflow section in README")

def main():
    """Main function to update README."""
    print("Updating README.md with data cleaning section...")
    add_cleaning_section()
    print("README.md update complete!")

if __name__ == "__main__":
    main()
