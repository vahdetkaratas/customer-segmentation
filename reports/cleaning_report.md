# Data Cleaning Report

Generated: 2025-08-28 03:33:27

## Summary
- **Total Rows**: 20
- **Invalid Values**: 0
- **Outliers**: 4
- **Exact Duplicates**: 0
- **Thin Customers**: 2

## Issues Found

### Missing Values

### Invalid Values

### Outliers
- **UnitPrice**: 4 outliers

### Date Range
- **Span**: 27 days (2025-07-31 07:56:05 to 2025-08-27 21:49:05)
- **Warning**: Span < 60 days

## Proposed Cleaning Plan
1. **winsorize_outliers**: Winsorize 4 outliers in UnitPrice

## Configuration
- **Outlier Method**: iqr
- **Outlier Handling**: winsorize
- **Drop Exact Duplicates**: True
- **Drop Thin Customers**: False
