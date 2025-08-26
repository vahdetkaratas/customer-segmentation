"""
Test script to verify RFM data scaling
Checks that scaled features have mean ≈ 0 and std ≈ 1
"""

import pandas as pd
import numpy as np

def test_scaling(file_path="data/processed/rfm_scaled.csv", tolerance=1e-3):
    """
    Test that scaled RFM data has correct statistical properties
    
    Parameters:
    - file_path: Path to the scaled RFM data CSV file
    - tolerance: Tolerance for mean and std deviation from expected values
    
    Returns:
    - Boolean indicating if test passed
    """
    print("Testing RFM data scaling...")
    
    try:
        # Load scaled data
        rfm_scaled = pd.read_csv(file_path)
        print(f"✓ Loaded scaled data: {len(rfm_scaled)} customers")
        
        # Get scaled features (exclude CustomerID)
        scaled_features = rfm_scaled[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
        
        # Test each feature
        test_passed = True
        expected_mean = 0.0
        expected_std = 1.0
        
        print("\nScaling Test Results:")
        print("-" * 40)
        
        for feature in scaled_features.columns:
            actual_mean = scaled_features[feature].mean()
            actual_std = scaled_features[feature].std()
            
            mean_ok = abs(actual_mean - expected_mean) < tolerance
            std_ok = abs(actual_std - expected_std) < tolerance
            
            status = "✅" if (mean_ok and std_ok) else "❌"
            
            print(f"{feature}:")
            print(f"  Mean: {actual_mean:.8f} (expected: {expected_mean:.1f}) {'✓' if mean_ok else '✗'}")
            print(f"  Std:  {actual_std:.8f} (expected: {expected_std:.1f}) {'✓' if std_ok else '✗'}")
            print(f"  Status: {status}")
            print()
            
            if not (mean_ok and std_ok):
                test_passed = False
        
        # Overall test result
        if test_passed:
            print("🎉 Scaling test passed ✅")
            print("All features have mean ≈ 0 and std ≈ 1 within tolerance")
        else:
            print("⚠️  Scaling test failed ❌")
            print("Some features do not meet the expected statistical properties")
            print("This may indicate an issue with the scaling process")
        
        return test_passed
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {file_path}")
        print("Please run the RFM scaling first (python src/rfm_scaling.py)")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def additional_checks(file_path="data/processed/rfm_scaled.csv"):
    """
    Perform additional checks on the scaled data
    
    Parameters:
    - file_path: Path to the scaled RFM data CSV file
    """
    print("\nAdditional Data Quality Checks:")
    print("-" * 40)
    
    try:
        rfm_scaled = pd.read_csv(file_path)
        
        # Check for missing values
        missing_values = rfm_scaled.isnull().sum()
        print(f"Missing values: {missing_values.sum()} total")
        
        # Check data types
        print(f"Data types: {rfm_scaled.dtypes.to_dict()}")
        
        # Check for infinite values
        infinite_values = np.isinf(rfm_scaled[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]).sum()
        print(f"Infinite values: {infinite_values.sum()} total")
        
        # Check value ranges
        scaled_features = rfm_scaled[['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']]
        print(f"Value ranges:")
        for feature in scaled_features.columns:
            min_val = scaled_features[feature].min()
            max_val = scaled_features[feature].max()
            print(f"  {feature}: [{min_val:.3f}, {max_val:.3f}]")
        
        print("✓ Additional checks completed")
        
    except Exception as e:
        print(f"❌ Error during additional checks: {str(e)}")

def main():
    """
    Main function to run all scaling tests
    """
    print("=" * 60)
    print("RFM SCALING TEST SUITE")
    print("=" * 60)
    
    # Run main scaling test
    test_result = test_scaling()
    
    # Run additional checks
    additional_checks()
    
    print("\n" + "=" * 60)
    if test_result:
        print("🎉 ALL TESTS PASSED - Scaling is working correctly!")
    else:
        print("⚠️  TESTS FAILED - Please check the scaling process")
    print("=" * 60)

if __name__ == "__main__":
    main()
