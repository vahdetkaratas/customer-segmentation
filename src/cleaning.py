#!/usr/bin/env python3
"""
Data Cleaning & Validation Library
Reusable functions for transaction data cleaning with two-phase workflow.
"""

import os
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for deterministic behavior."""
    np.random.seed(seed)
    logger.debug(f"Random seed set to {seed}")


def ensure_dirs(*paths: str) -> None:
    """Ensure directories exist, create if needed."""
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")


def load_config(config_path: str = "config/cleaning_rules.yml") -> Dict[str, Any]:
    """Load cleaning configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config: {e}")
        raise


def load_transactions(path: str) -> pd.DataFrame:
    """Load transaction data from CSV file."""
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {path}")
        raise


def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> Dict[str, Any]:
    """Validate DataFrame schema against expected columns."""
    issues = {
        'missing_columns': [],
        'extra_columns': [],
        'type_issues': []
    }
    
    # Check for missing columns
    missing = set(expected_columns) - set(df.columns)
    if missing:
        issues['missing_columns'] = list(missing)
        logger.warning(f"Missing columns: {missing}")
    
    # Check for extra columns
    extra = set(df.columns) - set(expected_columns)
    if extra:
        issues['extra_columns'] = list(extra)
        logger.info(f"Extra columns found: {extra}")
    
    # Check for null values in required columns
    for col in expected_columns:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            issues['type_issues'].append(f"{col}: {null_count} null values")
            logger.warning(f"{col}: {null_count} null values")
    
    return issues


def coerce_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Coerce data types and track failures."""
    df_clean = df.copy()
    issues = {
        'date_parse_failures': 0,
        'numeric_parse_failures': 0,
        'string_issues': 0
    }
    
    # Coerce InvoiceDate
    if 'InvoiceDate' in df_clean.columns:
        original_count = len(df_clean)
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')
        issues['date_parse_failures'] = original_count - df_clean['InvoiceDate'].notna().sum()
        if issues['date_parse_failures'] > 0:
            logger.warning(f"Failed to parse {issues['date_parse_failures']} dates")
    
    # Coerce numeric columns
    for col in ['Quantity', 'UnitPrice']:
        if col in df_clean.columns:
            original_count = len(df_clean)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            issues['numeric_parse_failures'] += original_count - df_clean[col].notna().sum()
    
    # Clean string columns
    for col in ['CustomerID', 'InvoiceNo']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            issues['string_issues'] += df_clean[col].str.contains(r'^\s+|\s+$').sum()
    
    logger.info(f"Type coercion completed. Issues: {issues}")
    return df_clean, issues


def detect_duplicates(df: pd.DataFrame, near_keys: List[str], drop_exact: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Detect exact and near duplicates."""
    stats = {
        'exact_duplicates': 0,
        'near_duplicates': 0
    }
    
    # Exact duplicates
    exact_dups = df.duplicated(keep='first')
    stats['exact_duplicates'] = exact_dups.sum()
    
    if drop_exact and stats['exact_duplicates'] > 0:
        df_clean = df[~exact_dups].copy()
        exact_dups_df = df[exact_dups].copy()
        logger.info(f"Removed {stats['exact_duplicates']} exact duplicates")
    else:
        df_clean = df.copy()
        exact_dups_df = df[exact_dups].copy() if stats['exact_duplicates'] > 0 else pd.DataFrame()
    
    # Near duplicates (same CustomerID, InvoiceNo, InvoiceDate)
    if all(key in df_clean.columns for key in near_keys):
        near_dups = df_clean.duplicated(subset=near_keys, keep='first')
        stats['near_duplicates'] = near_dups.sum()
        if stats['near_duplicates'] > 0:
            logger.warning(f"Found {stats['near_duplicates']} near duplicates")
    
    return df_clean, exact_dups_df, stats


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize series to specified quantiles."""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    
    winsorized = series.copy()
    winsorized[winsorized < lower_bound] = lower_bound
    winsorized[winsorized > upper_bound] = upper_bound
    
    return winsorized


def analyze_issues(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze data quality issues."""
    issues = {
        'total_rows': len(df),
        'missing_values': {},
        'invalid_values': {},
        'outliers': {},
        'date_span': {},
        'thin_customers': 0,
        'duplicates': {}
    }
    
    # Missing values
    for col in config['expected_columns']:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                issues['missing_values'][col] = missing_count
    
    # Invalid values
    if 'Quantity' in df.columns:
        invalid_qty = (df['Quantity'] <= 0).sum()
        if invalid_qty > 0:
            issues['invalid_values']['Quantity'] = invalid_qty
    
    if 'UnitPrice' in df.columns:
        invalid_price = (df['UnitPrice'] <= 0).sum()
        if invalid_price > 0:
            issues['invalid_values']['UnitPrice'] = invalid_price
    
    # Outliers
    for col in ['Quantity', 'UnitPrice']:
        if col in df.columns and df[col].notna().any():
            outliers = detect_outliers_iqr(df[col], config['outliers']['iqr_multiplier'])
            outlier_count = outliers.sum()
            if outlier_count > 0:
                issues['outliers'][col] = outlier_count
    
    # Date span
    if 'InvoiceDate' in df.columns and df['InvoiceDate'].notna().any():
        min_date = df['InvoiceDate'].min()
        max_date = df['InvoiceDate'].max()
        span_days = (max_date - min_date).days
        issues['date_span'] = {
            'min_date': str(min_date),
            'max_date': str(max_date),
            'span_days': span_days,
            'warning': span_days < config['date']['min_span_days']
        }
    
    # Thin customers
    if 'CustomerID' in df.columns:
        customer_invoices = df.groupby('CustomerID').size()
        thin_customers = (customer_invoices < config['customers']['flag_min_invoices']).sum()
        issues['thin_customers'] = int(thin_customers)
    
    # Duplicates
    exact_dups = df.duplicated().sum()
    issues['duplicates']['exact'] = int(exact_dups)
    
    if all(key in df.columns for key in config['duplicates']['near_keys']):
        near_dups = df.duplicated(subset=config['duplicates']['near_keys']).sum()
        issues['duplicates']['near'] = int(near_dups)
    
    logger.info(f"Analysis complete. Found {sum(issues['invalid_values'].values())} invalid values, {sum(issues['outliers'].values())} outliers")
    return issues


def propose_plan(issues: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Propose cleaning plan based on issues."""
    plan = []
    
    # Drop invalid quantities
    if issues['invalid_values'].get('Quantity', 0) > 0:
        plan.append({
            'step': 'drop_invalid_quantities',
            'target': 'Quantity',
            'reason': f"Remove {issues['invalid_values']['Quantity']} rows with Quantity <= 0",
            'count_estimate': issues['invalid_values']['Quantity']
        })
    
    # Drop invalid prices
    if issues['invalid_values'].get('UnitPrice', 0) > 0:
        plan.append({
            'step': 'drop_invalid_prices',
            'target': 'UnitPrice',
            'reason': f"Remove {issues['invalid_values']['UnitPrice']} rows with UnitPrice <= 0",
            'count_estimate': issues['invalid_values']['UnitPrice']
        })
    
    # Handle outliers
    for col, count in issues['outliers'].items():
        if count > 0:
            if config['outliers']['handle'] == 'winsorize':
                plan.append({
                    'step': 'winsorize_outliers',
                    'target': col,
                    'reason': f"Winsorize {count} outliers in {col}",
                    'count_estimate': count
                })
            elif config['outliers']['handle'] == 'drop':
                plan.append({
                    'step': 'drop_outliers',
                    'target': col,
                    'reason': f"Drop {count} outliers in {col}",
                    'count_estimate': count
                })
    
    # Drop exact duplicates
    if issues['duplicates'].get('exact', 0) > 0 and config['duplicates']['drop_exact']:
        plan.append({
            'step': 'drop_exact_duplicates',
            'target': 'all',
            'reason': f"Remove {issues['duplicates']['exact']} exact duplicates",
            'count_estimate': issues['duplicates']['exact']
        })
    
    # Drop thin customers (if configured)
    if issues['thin_customers'] > 0 and config['customers']['drop_thin_customers']:
        plan.append({
            'step': 'drop_thin_customers',
            'target': 'CustomerID',
            'reason': f"Remove customers with < {config['customers']['flag_min_invoices']} invoices",
            'count_estimate': issues['thin_customers']
        })
    
    logger.info(f"Proposed {len(plan)} cleaning steps")
    return plan


def apply_plan(df: pd.DataFrame, plan: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute cleaning plan and return cleaned, rejected, and issues dataframes."""
    df_clean = df.copy()
    rejected_rows = []
    issues_catalog = []
    
    for step in plan:
        step_name = step['step']
        logger.info(f"Executing: {step_name}")
        
        if step_name == 'drop_invalid_quantities':
            mask = df_clean['Quantity'] <= 0
            rejected = df_clean[mask].copy()
            rejected['issue_type'] = 'invalid_quantity'
            rejected_rows.append(rejected)
            df_clean = df_clean[~mask].copy()
            
        elif step_name == 'drop_invalid_prices':
            mask = df_clean['UnitPrice'] <= 0
            rejected = df_clean[mask].copy()
            rejected['issue_type'] = 'invalid_price'
            rejected_rows.append(rejected)
            df_clean = df_clean[~mask].copy()
            
        elif step_name == 'winsorize_outliers':
            col = step['target']
            outliers = detect_outliers_iqr(df_clean[col], config['outliers']['iqr_multiplier'])
            if outliers.any():
                # Create issues catalog entry
                outlier_rows = df_clean[outliers].copy()
                outlier_rows['issue_type'] = f'outlier_{col}'
                outlier_rows['original_value'] = outlier_rows[col]
                issues_catalog.append(outlier_rows)
                
                # Winsorize
                df_clean[col] = winsorize_series(df_clean[col], *config['outliers']['winsor_limits'])
                
        elif step_name == 'drop_outliers':
            col = step['target']
            outliers = detect_outliers_iqr(df_clean[col], config['outliers']['iqr_multiplier'])
            rejected = df_clean[outliers].copy()
            rejected['issue_type'] = f'outlier_{col}'
            rejected_rows.append(rejected)
            df_clean = df_clean[~outliers].copy()
            
        elif step_name == 'drop_exact_duplicates':
            duplicates = df_clean.duplicated(keep='first')
            rejected = df_clean[duplicates].copy()
            rejected['issue_type'] = 'exact_duplicate'
            rejected_rows.append(rejected)
            df_clean = df_clean[~duplicates].copy()
            
        elif step_name == 'drop_thin_customers':
            customer_invoices = df_clean.groupby('CustomerID').size()
            thin_customers = customer_invoices[customer_invoices < config['customers']['flag_min_invoices']].index
            mask = df_clean['CustomerID'].isin(thin_customers)
            rejected = df_clean[mask].copy()
            rejected['issue_type'] = 'thin_customer'
            rejected_rows.append(rejected)
            df_clean = df_clean[~mask].copy()
    
    # Combine rejected rows
    rejected_df = pd.concat(rejected_rows, ignore_index=True) if rejected_rows else pd.DataFrame()
    
    # Combine issues catalog
    issues_df = pd.concat(issues_catalog, ignore_index=True) if issues_catalog else pd.DataFrame()
    
    logger.info(f"Cleaning complete. Kept {len(df_clean)} rows, rejected {len(rejected_df)} rows")
    return df_clean, rejected_df, issues_df


def write_reports(issues: Dict[str, Any], plan: List[Dict[str, Any]], 
                 config: Dict[str, Any], samples: Dict[str, pd.DataFrame],
                 report_md_path: str, report_json_path: str) -> None:
    """Write cleaning reports in Markdown and JSON formats."""
    ensure_dirs(report_md_path, report_json_path)
    
    # JSON report
    json_report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'issues': issues,
        'plan': plan,
        'samples': {
            'invalid_quantities': samples.get('invalid_quantities', pd.DataFrame()).to_dict('records')[:5],
            'invalid_prices': samples.get('invalid_prices', pd.DataFrame()).to_dict('records')[:5],
            'outliers': samples.get('outliers', pd.DataFrame()).to_dict('records')[:5]
        }
    }
    
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, indent=2, default=str)
    
    # Markdown report
    md_content = f"""# Data Cleaning Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Rows**: {issues['total_rows']:,}
- **Invalid Values**: {sum(issues['invalid_values'].values()):,}
- **Outliers**: {sum(issues['outliers'].values()):,}
- **Exact Duplicates**: {issues['duplicates'].get('exact', 0):,}
- **Thin Customers**: {issues['thin_customers']:,}

## Issues Found

### Missing Values
"""
    
    for col, count in issues['missing_values'].items():
        md_content += f"- **{col}**: {count:,} missing values\n"
    
    md_content += "\n### Invalid Values\n"
    for col, count in issues['invalid_values'].items():
        md_content += f"- **{col}**: {count:,} invalid values\n"
    
    md_content += "\n### Outliers\n"
    for col, count in issues['outliers'].items():
        md_content += f"- **{col}**: {count:,} outliers\n"
    
    md_content += "\n### Date Range\n"
    if issues['date_span']:
        span = issues['date_span']
        md_content += f"- **Span**: {span['span_days']} days ({span['min_date']} to {span['max_date']})\n"
        if span['warning']:
            md_content += f"- **Warning**: Span < {config['date']['min_span_days']} days\n"
    
    md_content += "\n## Proposed Cleaning Plan\n"
    for i, step in enumerate(plan, 1):
        md_content += f"{i}. **{step['step']}**: {step['reason']}\n"
    
    md_content += "\n## Configuration\n"
    md_content += f"- **Outlier Method**: {config['outliers']['method']}\n"
    md_content += f"- **Outlier Handling**: {config['outliers']['handle']}\n"
    md_content += f"- **Drop Exact Duplicates**: {config['duplicates']['drop_exact']}\n"
    md_content += f"- **Drop Thin Customers**: {config['customers']['drop_thin_customers']}\n"
    
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"Reports written to {report_md_path} and {report_json_path}")
