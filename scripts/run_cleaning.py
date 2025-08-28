#!/usr/bin/env python3
"""
Data Cleaning CLI Tool
Two-phase workflow: ANALYZE (dry-run) and APPLY (execute cleaning).
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append('src')

from cleaning import (
    set_seed, ensure_dirs, load_config, load_transactions,
    validate_schema, coerce_types, analyze_issues, propose_plan,
    apply_plan, write_reports
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_mode(input_path: str, config_path: str, preview: int = 10, 
                save_plan: str = None, strict: bool = False) -> int:
    """Run analysis mode (dry-run)."""
    logger.info("Starting ANALYZE mode (dry-run)")
    
    try:
        # Load data and config
        config = load_config(config_path)
        df = load_transactions(input_path)
        
        # Validate schema
        schema_issues = validate_schema(df, config['expected_columns'])
        if schema_issues['missing_columns'] and strict:
            logger.error(f"Missing required columns: {schema_issues['missing_columns']}")
            return 1
        
        # Coerce types (non-destructive)
        df_clean, type_issues = coerce_types(df)
        
        # Analyze issues
        issues = analyze_issues(df_clean, config)
        
        # Propose cleaning plan
        plan = propose_plan(issues, config)
        
        # Collect sample data for report
        samples = {}
        if issues['invalid_values'].get('Quantity', 0) > 0:
            samples['invalid_quantities'] = df_clean[df_clean['Quantity'] <= 0].head(preview)
        if issues['invalid_values'].get('UnitPrice', 0) > 0:
            samples['invalid_prices'] = df_clean[df_clean['UnitPrice'] <= 0].head(preview)
        
        # Write reports
        write_reports(
            issues, plan, config, samples,
            config['outputs']['dryrun_report_md'],
            config['outputs']['dryrun_report_json']
        )
        
        # Save plan if requested
        if save_plan:
            ensure_dirs(save_plan)
            with open(save_plan, 'w', encoding='utf-8') as f:
                json.dump(plan, f, indent=2, default=str)
            logger.info(f"Plan saved to {save_plan}")
        
        # Print summary
        print(f"\nANALYSIS COMPLETE")
        print(f"Input: {input_path}")
        print(f"Total rows: {issues['total_rows']:,}")
        print(f"Invalid values: {sum(issues['invalid_values'].values()):,}")
        print(f"Outliers: {sum(issues['outliers'].values()):,}")
        print(f"Duplicates: {issues['duplicates'].get('exact', 0):,}")
        print(f"Thin customers: {issues['thin_customers']:,}")
        print(f"Proposed steps: {len(plan)}")
        print(f"Reports: {config['outputs']['dryrun_report_md']}")
        
        if strict and (sum(issues['invalid_values'].values()) > 0 or 
                      schema_issues['missing_columns']):
            logger.error("Strict mode: Found critical issues")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


def apply_mode(input_path: str, config_path: str, plan_from: str = None, 
               strict: bool = False) -> int:
    """Run apply mode (execute cleaning)."""
    logger.info("Starting APPLY mode")
    
    try:
        # Load data and config
        config = load_config(config_path)
        df = load_transactions(input_path)
        
        # Load or generate plan
        if plan_from:
            logger.info(f"Loading plan from {plan_from}")
            with open(plan_from, 'r', encoding='utf-8') as f:
                plan = json.load(f)
        else:
            logger.info("Generating fresh plan")
            schema_issues = validate_schema(df, config['expected_columns'])
            df_clean, _ = coerce_types(df)
            issues = analyze_issues(df_clean, config)
            plan = propose_plan(issues, config)
        
        # Apply plan
        df_clean, _ = coerce_types(df)
        cleaned_df, rejected_df, issues_df = apply_plan(df_clean, plan, config)
        
        # Ensure output directories
        ensure_dirs(
            config['outputs']['cleaned_csv'],
            config['outputs']['rejected_csv'],
            config['outputs']['issues_csv']
        )
        
        # Write outputs
        cleaned_df.to_csv(config['outputs']['cleaned_csv'], index=False)
        logger.info(f"Cleaned data saved to {config['outputs']['cleaned_csv']}")
        
        if len(rejected_df) > 0:
            rejected_df.to_csv(config['outputs']['rejected_csv'], index=False)
            logger.info(f"Rejected data saved to {config['outputs']['rejected_csv']}")
        else:
            # Create empty file
            rejected_df.to_csv(config['outputs']['rejected_csv'], index=False)
        
        if len(issues_df) > 0:
            issues_df.to_csv(config['outputs']['issues_csv'], index=False)
            logger.info(f"Issues catalog saved to {config['outputs']['issues_csv']}")
        else:
            # Create empty file
            issues_df.to_csv(config['outputs']['issues_csv'], index=False)
        
        # Print summary
        print(f"\nCLEANING COMPLETE")
        print(f"Input: {input_path}")
        print(f"Kept: {len(cleaned_df):,} rows")
        print(f"Rejected: {len(rejected_df):,} rows")
        print(f"Issues catalog: {len(issues_df):,} rows")
        print(f"Outputs:")
        print(f"   - Cleaned: {config['outputs']['cleaned_csv']}")
        print(f"   - Rejected: {config['outputs']['rejected_csv']}")
        print(f"   - Issues: {config['outputs']['issues_csv']}")
        
        # Validate cleaned data
        if len(cleaned_df) == 0:
            logger.error("No rows remaining after cleaning")
            return 1
        
        # Check for invalid values in cleaned data
        invalid_qty = (cleaned_df['Quantity'] <= 0).sum() if 'Quantity' in cleaned_df.columns else 0
        invalid_price = (cleaned_df['UnitPrice'] <= 0).sum() if 'UnitPrice' in cleaned_df.columns else 0
        
        if invalid_qty > 0 or invalid_price > 0:
            logger.error(f"Cleaned data still contains invalid values: {invalid_qty} quantities, {invalid_price} prices")
            if strict:
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Cleaning failed: {e}")
        return 1


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Data Cleaning Tool - Analyze first, apply on approval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze (dry-run)
  python scripts/run_cleaning.py --input data/sample/online_retail_sample.csv
  
  # Apply after approval
  python scripts/run_cleaning.py --input data/sample/online_retail_sample.csv --apply
  
  # Apply with specific plan
  python scripts/run_cleaning.py --input data/sample/online_retail_sample.csv --apply --plan-from path/to/plan.json
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/sample/online_retail_sample.csv',
        help='Input CSV file path (default: data/sample/online_retail_sample.csv)'
    )
    
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Execute cleaning plan (default: analyze only)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/cleaning_rules.yml',
        help='Configuration file path (default: config/cleaning_rules.yml)'
    )
    
    parser.add_argument(
        '--preview',
        type=int,
        default=10,
        help='Number of preview rows in report (default: 10)'
    )
    
    parser.add_argument(
        '--save-plan',
        type=str,
        help='Save generated plan to JSON file'
    )
    
    parser.add_argument(
        '--plan-from',
        type=str,
        help='Load plan from JSON file (for apply mode)'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors and exit non-zero'
    )
    
    args = parser.parse_args()
    
    # Set seed for deterministic behavior
    set_seed(42)
    
    # Validate input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Run appropriate mode
    if args.apply:
        return apply_mode(args.input, args.config, args.plan_from, args.strict)
    else:
        return analyze_mode(args.input, args.config, args.preview, args.save_plan, args.strict)


if __name__ == "__main__":
    sys.exit(main())
